"""topk：确定性流式 Top-K 内核。

对外入口：
    Entry       一条候选记录：key、score、seq 与延迟删除标记
    StreamTopK  有界最小堆：同分淘汰、延迟删除与统计计数
"""

from .core import Entry, StreamTopK

__all__ = ["Entry", "StreamTopK"]
