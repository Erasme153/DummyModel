#!/usr/bin/env python3
"""为 M1 英文预训练准备本地 FineWeb-Edu 数据，不需要联网。

处理顺序：轮询多个抓取批次 -> 轻量清洗 -> 扫描范围内精确去重 -> 按文档划分
train/validation -> Mistral 32K 编码并追加 EOS -> 分别拼接成定长序列。

只处理达到 token 预算所需的文本，不把几百 GB 原始数据一次性加载到内存。
输出是小端 uint16 的 train.bin / validation.bin，每行逻辑上为 2048 个 token；
格式、统计和完成状态写入 summary.json。只有 status=complete 才可用于训练。

阅读时可以把流程分成三层：
1. iter_sample_batches / iter_dump_batches：决定按什么顺序读取哪些文档。
2. normalize_text / split_document：决定哪些正文可用、属于哪个数据集。
3. PackedWriter：只接收整数 token ID，把多个文档拼成模型需要的定长序列。

这里有两种不同的 batch：读取 batch 是若干篇长度不一的文档；训练 batch 是
若干条长度相同的 token 序列。--batch-size 控制前者，不是训练时的 batch size。
“2048-token 序列”也不是一篇文章：它可以包含多篇短文，或只包含长文的一部分。
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import unicodedata

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# NumPy dtype 的 "<" 表示小端字节序，"u" 表示无符号整数，"2" 表示每个 ID
# 占 2 字节，可存 0～65535。32K 词表可以装下，磁盘占用为 int64 的四分之一。
# 这是存储格式；训练时仍应转换为 torch.long，二进制文件中也不保存行分隔符。
TOKEN_DTYPE = np.dtype("<u2")

# r"..." 是 Python 原始字符串，让反斜杠原样交给正则引擎；re.compile 则把
# 表达式预编译并复用，避免每处理一篇文档都重新构建同一个匹配器。
#
# r"[^\S\n]+" 逐项理解：
#   [...]  是字符集合；开头的 ^ 表示“排除集合中的字符”；
#   \S     是“非空白字符”；\n 是换行符；
#   [^\S\n] 因此匹配“属于空白、但不是 \n”的一个字符；
#   +      表示连续匹配一个或多个，把一整段空白视为一次匹配。
# 后面用 sub(" ", ...) 将空格、tab 等连续空白压成一个普通空格：
# "hello\t  world" -> "hello world"。它也能匹配 Unicode 空白，不仅是 ASCII 空格。
# 不直接用 r"\s+"，是为了保留换行所表达的段落结构。
HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")

# r"\n{3,}" 中 {3,} 表示至少重复 3 次：只匹配三个及以上连续换行。
# 后面替换成两个换行："A\n\n\n\nB" -> "A\n\nB"。
# 两个换行表示段落之间保留一个空行；单个换行和已有的两个换行不受影响。
EXTRA_NEWLINES = re.compile(r"\n{3,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default="/diff/workspace/wyl/data/fineweb-edu",
                        help="已下载的 fineweb-edu 仓库根目录，内含 data/CC-MAIN-*/*.parquet")
    parser.add_argument("--tokenizer", type=Path, default="/diff/models/mistralai/Mistral-7B-v0.1/tokenizer.json",
                        help="本地 Mistral-7B-v0.1/tokenizer.json")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data/tokenized/m01_fineweb_100m")
    parser.add_argument("--train-tokens", type=int, default=100_000_000)
    parser.add_argument("--validation-tokens", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.01,
                        help="文档被划为验证集的概率；不是精确的 token 比例")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="每次从一个抓取批次读取并编码的文档数")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log-every", type=int, default=50, help="每多少个读取 batch 打印进度")
    args = parser.parse_args()
    # 长度为 1 的序列经过 next-token 对齐后没有预测目标，所以至少需要两个 token。
    # 两个 split 都必须容纳完整一行；否则后面的预算向下取整会让其中一个变成空集。
    if args.sequence_length < 2:
        parser.error("--sequence-length 必须至少为 2，以计算 next-token loss")
    if min(args.train_tokens, args.validation_tokens) < args.sequence_length:
        parser.error("训练和验证 token 预算都必须至少容纳一个完整序列")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction 必须位于 (0, 1)")
    if not 0 < args.min_chars <= args.max_chars:
        parser.error("字符范围必须满足 0 < min-chars <= max-chars")
    if min(args.batch_size, args.log_every) <= 0:
        parser.error("--batch-size 和 --log-every 必须为正数")
    return args


def normalize_text(value: object, min_chars: int, max_chars: int) -> tuple[str | None, str | None]:
    """轻量规范化并拒绝明显异常文本；这些规则不是完整的质量分类器。

    NFC 合并等价的 Unicode 字符，统一换行和空白，但保留大小写、标点和段落。
    FineWeb-Edu 上游已经做过英文与教育质量筛选，这里不重复复杂的质量打分。

    返回 (正文, None) 表示通过；返回 (None, 原因) 表示丢弃。原因保留为计数项，
    方便事后判断规则是否过严。清洗先于去重，这样只差空格或换行格式的正文才会
    得到相同指纹。min_chars / max_chars 计算的是 Unicode 字符数，不是 token 数。
    """
    if not isinstance(value, str):
        return None, "not_text"
    # 先挡住超长原文，减少后续逐字符处理的开销。即使它清理空白后可能变短，
    # 本规则仍会直接拒绝它；这是明确的过滤取舍，而不是截断后继续使用。
    if len(value) > max_chars:
        return None, "too_long"

    # Unicode 类别 Cc 是控制字符，如 NUL。正常排版用的换行、回车和 tab 除外。
    # any() 发现第一个异常就停止检查；这里拒绝整篇，不悄悄删除异常字符拼接正文。
    if any(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in value):
        return None, "control_character"
    # U+FFFD（�）通常是解码失败时填入的替换字符。此处采用较保守的规则：含有
    # 一个就过滤，因此也可能舍弃本来就在讨论该符号的正常文章，计数会记录下来。
    if "\ufffd" in value:
        return None, "replacement_character"

    # NFC 会把 e + 组合重音符合成 é，使视觉上等价的文字更容易被精确去重。
    # 先把 Windows 的 CRLF 换成 LF，再处理单独的 CR；顺序反过来会把 CRLF
    # 变成两个 LF，错误地产生空段落。这里不用会额外折叠兼容字符的 NFKC。
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    # U+FEFF 是常见的 BOM 字符；在已经解码的正文中去掉它，避免不可见字符影响指纹。
    text = text.replace("\ufeff", "")
    # split 保留行结构；sub 压缩每行内部的空白；strip 去除行首尾空白；最后
    # join 恢复换行。原来只含空格的行变成空行，而不会与前后段落直接黏在一起。
    text = "\n".join(HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    # 行内空白处理后，再压缩连续空行并去掉整篇首尾空白，去重使用这个最终正文。
    text = EXTRA_NEWLINES.sub("\n\n", text).strip()

    # 清理后重新检查长度：原文可能主要是空格，不能只依赖清理前的长度检查。
    if len(text) < min_chars:
        return None, "too_short"
    if len(text) > max_chars:
        return None, "too_long"
    # bool 在求和时 True 算 1，所以这里是在计算“字母数 / 正文总字符数”。
    # 分母包括空格、标点和换行；比例过低的数字表格或符号噪声会被过滤。
    # isalpha() 识别 Unicode 字母，并非只认 A～Z，因此本规则不能用于识别英语。
    if sum(ch.isalpha() for ch in text) / len(text) < 0.5:
        return None, "low_letter_fraction"
    return text, None


def split_document(digest: bytes, seed: int, validation_fraction: float) -> str:
    """按规范化文本的指纹分组，划分结果不依赖文档顺序或出现在哪个文件。

    不能使用 Python 的 hash()，它在不同进程之间可能变化。在 seed 和划分比例
    不变时，对同样的正文始终给出同一 split；即使以后扩充原始分片，重复文档
    也不会跨 train/validation。这不代表扩充文件后入选文档或训练顺序不变。
    这只防止规范化后的完全重复，不保证近似重复、同网站文章或段落不重合。
    """
    # digest 是正文 SHA-256 的 32 字节结果。把种子与它再次做哈希，相当于为
    # 每篇文档生成一个稳定的“抽签值”；改变 seed 可以得到另一套划分。
    value = hashlib.sha256(str(seed).encode("ascii") + b":" + digest).digest()
    # 取前 8 字节解释为 64 位无符号整数，再除以 2**64，缩放后与概率阈值比较。
    # "big" 只决定怎样解释这 8 字节，与 token 文件的小端存储格式没有关系。
    # 例如阈值 0.01，抽签值 0.006 进入验证集，0.72 进入训练集。
    # 这让验证文档数在统计上接近 1%，不保证精确 1%，更不保证验证 token 占比。
    fraction = int.from_bytes(value[:8], "big") / 2**64
    return "validation" if fraction < validation_fraction else "train"


def iter_dump_batches(paths: list[Path], root: Path, seed: str, batch_size: int,
                      sources: dict):
    """逐个打开同一抓取批次的分片；打乱分片、row group 和 batch 内文档顺序。

    Parquet 的 row group 可以独立读取，因此不用从每个大文件的开头顺序扫描。
    只读 text/id/url 三列。读取或格式检查失败时直接报错，不把坏分片静默跳过。

    三个层次不要混淆：一个 Parquet 文件包含多个 row group（行组）；一个行组
    又可分成多个读取 batch。下面打乱的是行组顺序及 batch 内顺序，没有把整个
    文件的所有行展开后做全局 shuffle。yield 每次交出 (来源, 文档列表)，随后
    暂停，等调用者下一次 next() 时继续读取。
    """
    # 独立的随机数生成器不影响训练或其他抓取批次的随机状态。先排序再 shuffle，
    # 消除文件系统枚举顺序的影响；相同输入列表和 seed 才能复现同样的读取顺序。
    rng = random.Random(seed)
    paths = sorted(paths)
    rng.shuffle(paths)
    for path in paths:
        relative = path.relative_to(root).as_posix()
        try:
            with pq.ParquetFile(path) as parquet:
                schema = parquet.schema_arrow
                # Arrow 的 string / large_string 都是字符串类型，主要区别是内部
                # 使用的偏移量宽度。text 是必需列，id/url 只用于追溯，可以缺省。
                if "text" not in schema.names or not (
                    pa.types.is_string(schema.field("text").type)
                    or pa.types.is_large_string(schema.field("text").type)
                ):
                    raise ValueError("缺少字符串类型的 text 列")
                # 记录刚开始读取时的大小和修改时间，结束时再比较，检查处理中源文件
                # 是否变动。只读元数据，不为几个 GB 的分片再计算一遍全文哈希。
                stat = path.stat()
                sources[relative] = {
                    "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "rows_in_file": parquet.metadata.num_rows,
                    "row_groups": parquet.num_row_groups,
                }
                groups = list(range(parquet.num_row_groups))
                rng.shuffle(groups)
                # Parquet 是列式格式，指定 columns 可以不读取质量分数等未使用的列。
                # use_threads=False 关闭 Arrow 的多线程解码；tokenizer 会单独并行。
                columns = [name for name in ("text", "id", "url") if name in schema.names]
                for batch in parquet.iter_batches(batch_size=batch_size, columns=columns,
                                                  row_groups=groups, use_threads=False):
                    # 仅把当前 batch 转为 Python 字典列表，不把整个文件搬进内存。
                    rows = batch.to_pylist()
                    rng.shuffle(rows)
                    yield relative, rows
        except Exception as exc:
            raise RuntimeError(f"读取 Parquet 失败：{path}") from exc


def iter_sample_batches(files: list[Path], root: Path, seed: int, batch_size: int,
                        sources: dict):
    """在已有 CC-MAIN 批次之间轮询，避免只用最早一个分片就填满预算。

    每个抓取批次同时只打开一个分片；当前十个批次最多对应十个读取器。
    这是按本地抓取批次轮询的教学采样，不是按全库文档数加权的均匀随机采样。

    例如有 A、B、C 三个批次，依次读取 A 的一个 batch、B 的一个 batch、C 的
    一个 batch，再回到 A。清洗通过率和文档长度不同，最终 token 贡献不会相等。
    """
    by_dump: dict[str, list[Path]] = {}
    for path in files:
        by_dump.setdefault(path.parent.name, []).append(path)
    dumps = sorted(by_dump)
    random.Random(seed).shuffle(dumps)
    # deque 是两端都能高效插入/弹出的队列。队列里放的是尚未读完的生成器，
    # 创建生成器本身不会立刻读取文件；各批次还带有独立且可复现的派生种子。
    readers = deque(iter_dump_batches(by_dump[dump], root, f"{seed}:{dump}", batch_size, sources)
                    for dump in dumps)
    try:
        while readers:
            reader = readers.popleft()
            try:
                item = next(reader)
            except StopIteration:
                # 该批次所有分片均读完，不再放回队列；其余批次继续轮询。
                continue
            except BaseException:
                # 这个读取器已被弹出，finally 中看不到它，故此处单独关闭再抛错。
                # BaseException 还覆盖 Ctrl+C；这里只做清理，不吞掉中断或读取异常。
                reader.close()
                raise
            # 把还有数据的读取器放回队尾，下一轮轮到其他批次。
            readers.append(reader)
            yield item
    finally:
        # 上层可能在预算满足时提前结束，生成器也可能暂停在 Parquet 的 with 内。
        # 显式 close() 会让它们退出 with，及时释放未读完文件的资源。
        for reader in readers:
            reader.close()


class PackedWriter:
    """把一个 split 的连续 token 流写成定长行，内存中只保留不足一行的尾巴。

    train 与 validation 必须各有一个实例，绝不共享拼接缓存。预算向下取整到
    sequence_length 的整数倍；达到预算时最后一篇文档可能被截断。不添加 padding，
    不循环复制文本补齐数据。EOS 区分文档，但这里不做跨文档 attention 隔离。
    """

    def __init__(self, handle, requested_tokens: int, sequence_length: int):
        self.handle = handle
        self.sequence_length = sequence_length
        # 只存完整行。例如请求 10 tokens、行长 4，实际目标是 (10 // 4) * 4 = 8。
        # 这样不用为最后不足一行的部分引入 padding、padding mask 和忽略 loss 的逻辑。
        self.target_tokens = requested_tokens // sequence_length * sequence_length
        # written_tokens 只统计已写入文件的整行；pending 是已收下但还未满一行的尾巴。
        # 两者之和才是目前已占用的 token 预算，且始终有 len(pending) < 行长。
        self.written_tokens = 0
        self.pending: list[int] = []
        self.documents = 0
        self.truncated_tokens = 0

    @property
    def full(self) -> bool:
        # 目标本身是行长的整数倍，达到它时 pending 必然为空，不会漏掉已接收的尾巴。
        return self.written_tokens == self.target_tokens

    def add(self, ids: list[int]) -> int:
        """接收一篇文档的 token（已含 EOS），返回实际纳入本 split 的 token 数。

        例：行长 4，上一篇留下 [a, b, EOS]，新文档是 [c, d, EOS]。
        拼接后写出 [a, b, EOS, c]，留下 [d, EOS] 等下一篇；不会把 EOS 当 padding，
        也不会要求每篇文章从行首开始。这里只切序列，labels 的 shift 留给模型。
        """
        # 尾巴也已占预算，必须减掉；否则每来一篇都可能多收 len(pending) 个 token。
        remaining = self.target_tokens - self.written_tokens - len(self.pending)
        used = min(remaining, len(ids))
        if used == 0:
            return 0
        tokens = self.pending + ids[:used]
        # complete 是可以落盘的 token 数，而不是行数。例如合计 11 个 token，
        # 行长 4，则写入前 8 个，余下 3 个暂存。bytes 中只是一串 ID，不需要 reshape。
        complete = len(tokens) // self.sequence_length * self.sequence_length
        if complete:
            self.handle.write(np.asarray(tokens[:complete], dtype=TOKEN_DTYPE).tobytes())
            self.written_tokens += complete
        self.pending = tokens[complete:]
        self.documents += 1
        # 如果在总预算边界截断了一篇文档，舍弃部分（可能包含 EOS）只计入这里。
        # pending 不算舍弃：它会在下一篇到来时继续拼接，除非语料提前耗尽。
        self.truncated_tokens += len(ids) - used
        return used

    def stats(self) -> dict:
        # unused_tail_tokens 在正常达到预算时为 0；若源数据耗尽，它可能非零，且
        # 不在 written_tokens 或文件字节数中。documents 则统计所有贡献过 token 的文档。
        return {
            "target_tokens": self.target_tokens,
            "written_tokens": self.written_tokens,
            "sequences": self.written_tokens // self.sequence_length,
            "documents": self.documents,
            "unused_tail_tokens": len(self.pending),
            "truncated_tokens_at_budget": self.truncated_tokens,
            "bytes": self.written_tokens * TOKEN_DTYPE.itemsize,
        }


def write_json_line(handle, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    # 明确只用正式 data/ 分片，排除 sample/ 中与全量语料重叠的抽样副本和断点文件。
    files = sorted(root.glob("data/CC-MAIN-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"没有找到 {root}/data/CC-MAIN-*/*.parquet")
    # 限制 Rust tokenizer 的线程数，避免默认占满宿主机 CPU；用户可在命令前覆盖。
    # setdefault 只在当前进程还没设置该变量时给默认值，不写入 Conda 或 shell 配置。
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if tokenizer.get_vocab_size() != 32_000:
        raise ValueError("本轮固定使用 32K tokenizer，请检查 --tokenizer")
    # 词表大小是“共有多少项”，不等于最大 ID；存入 uint16 前还要核对 ID 的上界，
    # 避免超出范围的整数在类型转换时被截断。这里不按词表大小猜测特殊 token 的 ID。
    if max(tokenizer.get_vocab().values()) > np.iinfo(TOKEN_DTYPE).max:
        raise ValueError("token ID 超出 uint16 存储范围")
    eos = tokenizer.token_to_id("</s>")
    if eos is None:
        raise ValueError("tokenizer 缺少 Mistral EOS </s>")
    # 导入的 tokenizer.json 可能带有原模型的 padding/truncation 配置。预训练先
    # 编码完整文档，再由 PackedWriter 按本轮长度处理，不能在编码时先截掉文档尾部。
    # 这两项与 add_special_tokens=False 各管一件事，不会互相替代。
    tokenizer.no_padding()
    tokenizer.no_truncation()

    # 防止误覆盖一次成功实验。重新准备时使用新的输出目录，不提供强制覆盖开关。
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer.save(str(output / "tokenizer.json"))
    # monotonic 是持续递增的计时钟，统计耗时不会受系统日期校准的影响。
    start = time.monotonic()
    # Counter 访问尚未出现的键时返回 0，便于在各个分支直接累加统计。
    # counts 记处理流程，filtered 记拒绝原因，selected_by_source 记各来源最终贡献。
    counts: Counter = Counter()
    filtered: Counter = Counter()
    selected_by_source: dict[str, Counter] = {}
    sources: dict = {}
    # seen 位于所有 batch / 抓取批次之外，因此去重范围覆盖本次扫描过的所有来源。
    # 仅存 32 字节指纹，不存全文；集合内存仍随唯一文档数增长，不是常量内存。
    # 此实现适合当前约 9 万篇的实验，不能据此认为它可以无限扩展到全库去重。
    seen: set[bytes] = set()
    preview_counts: Counter = Counter()

    # ExitStack 统一管理运行中打开的多个文件和生成器。无论正常结束、读取异常
    # 还是 Ctrl+C，退出时都会按逆序释放资源，先关读取器，再关闭并刷出输出文件。
    with ExitStack() as stack:
        # 两个 writer 拥有各自的 token 预算和 pending，训练文档的尾巴不可能被
        # 拼到验证文档前面。若先统一 packing 再切 split，就无法保证这一点。
        writers = {
            split: PackedWriter(stack.enter_context((output / f"{split}.bin").open("wb")),
                                budget, args.sequence_length)
            for split, budget in (("train", args.train_tokens), ("validation", args.validation_tokens))
        }
        documents = stack.enter_context((output / "documents.jsonl").open("w", encoding="utf-8"))
        previews = stack.enter_context((output / "preview.jsonl").open("w", encoding="utf-8"))
        batches = iter_sample_batches(files, root, args.seed, args.batch_size, sources)
        stack.callback(batches.close)
        print(f"Found {len(files)} local shards. Output: {output}", flush=True)
        for batch_index, (source, rows) in enumerate(batches, 1):
            # 先筛出当前 batch 的可用文档，再一次性交给 Rust tokenizer 批量编码。
            # 每个候选同时保存正文、指纹、split、原始元信息，编码后仍可一一对应。
            candidates = []
            for row in rows:
                counts["scanned_documents"] += 1
                text, reason = normalize_text(row["text"], args.min_chars, args.max_chars)
                if reason is not None:
                    filtered[reason] += 1
                    continue
                # 指纹基于清洗后的正文，而不是 URL 或网页 ID。同一正文即使来自
                # 不同网址也会判重；同一网址若正文不同，则不会仅因为 URL 相同被去掉。
                # digest() 返回二进制指纹用于 set；写索引时才转成可读的十六进制。
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest in seen:
                    counts["duplicate_documents"] += 1
                    continue
                seen.add(digest)
                split = split_document(digest, args.seed, args.validation_fraction)
                # 一个 split 先满时，仍要继续寻找另一个 split 的文档。不能把后续
                # train 文档改分到 validation 凑数，否则会破坏稳定的文档划分规则。
                # 此处提前跳过，也省去对这篇文章的编码成本。
                if writers[split].full:
                    counts["split_quota_skipped"] += 1
                    continue
                candidates.append((text, digest, split, row))

            # add_special_tokens=False 跳过 tokenizer 后处理器自动插入的 BOS/EOS；
            # 这并不禁止正文中的特殊 token 字面串被识别。下文只显式追加一个 EOS。
            # encode_batch 保持输入顺序，所以可以用 zip 将每个编码结果配回原文档。
            encodings = tokenizer.encode_batch([item[0] for item in candidates], add_special_tokens=False)
            for (text, digest, split, row), encoding in zip(candidates, encodings):
                writer = writers[split]
                # 第二次检查不是重复工作：同一 batch 的早先文档可能刚好填满预算，
                # 而 candidates 是在处理整个 batch 之前生成的，里面仍有该 split 的候选。
                if writer.full:
                    counts["split_quota_skipped"] += 1
                    continue
                ids = encoding.ids + [eos]
                # offset 的单位是 token，不是字节，并且相对于各 split 自己的 token 流。
                # 文档可能从尚未写出的 pending 后开始，因此不能只看 written_tokens。
                offset = writer.written_tokens + len(writer.pending)
                used = writer.add(ids)
                record = {
                    "split": split, "sha256": digest.hex(), "source": source,
                    "id": row.get("id"), "url": row.get("url"), "characters": len(text),
                    "encoded_tokens_with_eos": len(ids), "used_tokens": used, "token_offset": offset,
                }
                # encoded_tokens_with_eos 是原文完整编码长度；used_tokens 是预算内接收
                # 的长度。最后一篇可能只用前半段，二者不能混着统计实际训练数据量。
                # 记录正文指纹与来源，便于核对重复和 train/validation 重叠；不再复制全部正文。
                write_json_line(documents, record)
                selected_by_source.setdefault(source, Counter())[f"{split}_documents"] += 1
                selected_by_source[source][f"{split}_tokens"] += used
                # 预览只取每个 split 前三篇的一小段，供人工检查清洗结果；这不是另一份
                # 训练文本，也不能仅凭六条预览就声称全体数据没有质量问题。
                if preview_counts[split] < 3:
                    write_json_line(previews, {**record, "text_preview": text[:1200]})
                    preview_counts[split] += 1
            if batch_index % args.log_every == 0:
                print(f"batches={batch_index} scanned={counts['scanned_documents']:,} "
                      f"train={writers['train'].written_tokens:,}/{writers['train'].target_tokens:,} "
                      f"validation={writers['validation'].written_tokens:,}/"
                      f"{writers['validation'].target_tokens:,}", flush=True)
            # 训练集和验证集的 token 预算都满足才停止；哈希划分的文档长度不同，
            # 即使文档比例接近预设值，两边也不一定在同一个读取 batch 内达到目标。
            if all(writer.full for writer in writers.values()):
                break

    # 文件关闭后才写完成标记；中途异常或 Ctrl+C 不会留下 status=complete 的 summary。
    complete = all(writer.full for writer in writers.values())
    # 大小和 mtime 是廉价的一致性检查，不等于对原始文件做了完整内容校验。
    for source, metadata in sources.items():
        stat = (root / source).stat()
        if (stat.st_size, stat.st_mtime_ns) != (metadata["bytes"], metadata["mtime_ns"]):
            raise RuntimeError(f"准备期间原始文件发生变化：{source}")
    # discovered_files 是启动时发现的分片数，files_read 是实际打开过的分片。
    # 达到预算就提前结束，因此前者可能远大于后者；complete 表示本轮预算已满足，
    # 不表示整套 FineWeb-Edu 已经下载、扫描或去重完成。
    # tokenizer 的 SHA-256 则针对本次实际使用的副本，便于训练时核对词表是否一致。
    summary = {
        "status": "complete" if complete else "insufficient_data",
        "source": {"repo_id": "HuggingFaceFW/fineweb-edu", "license_in_dataset_card": "odc-by",
                   "input_dir": str(root), "discovered_files": len(files),
                   "sampling": "seeded per-dump round-robin; shuffled files, row groups and batches",
                   "files_read": sources, "selected_by_source": selected_by_source},
        "parameters": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        "tokenizer": {"source_path": str(args.tokenizer.resolve()), "file": "tokenizer.json",
                      "sha256": hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest(),
                      "vocab_size": tokenizer.get_vocab_size(), "eos_token_id": eos,
                      "add_bos": False, "append_eos": True},
        "format": {"dtype": "<u2", "sequence_length": args.sequence_length,
                   "packing": "separate continuous streams per split; no padding or document mask",
                   "labels": "pass input_ids as labels; MiniLlama shifts labels internally"},
        "counts": dict(counts), "filtered_documents": dict(filtered),
        "splits": {split: writer.stats() for split, writer in writers.items()},
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"status": summary["status"], "splits": summary["splits"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    # 文件存在甚至能 reshape，并不表示预算达标。语料耗尽时保留统计供排查，
    # 同时以异常退出，让调用脚本或使用者知道这次没有准备成功。
    if not complete:
        raise RuntimeError("原始数据耗尽，未达到预算；请检查 summary.json，不能当作已完成的数据集")


if __name__ == "__main__":
    main()
