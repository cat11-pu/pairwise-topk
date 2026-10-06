"""topk：确定性流式 Top-K 内核（有界最小堆 + 同分淘汰 + 延迟删除）。

组件：
    Entry       一条候选记录：key、score、seq 与延迟删除标记
    StreamTopK  有界最小堆内核：堆顶是最弱候选，也就是下一个被替换的对象

约定：
    * 分数是有限实数，越大越强；同分时先到的候选胜出，同分的新来者不得挤掉在位候选；
    * 结果最多 k 条；进入结果的不同 key 不足 k 条时按实际条数返回，不补空位；
    * 同一 key 重复入流按最后一次的分数更新，seq 与席位都保持不变；
    * discard 只打延迟删除标记：标记后立刻不再是结果，堆数组里的空槽留到后续
      取堆顶时惰性回收，因此这一步的开销与 k 无关；
    * 索引与堆数组共用同一条记录：标记为删除的记录在物理回收之前仍可能留在两处，
      物理回收时只摘掉仍然指向该条记录的映射；
    * 全程只依赖调用参数与内部状态，不读时钟、不起线程、不做任何 I/O。
"""

__all__ = ["Entry", "StreamTopK"]


def _normalize_score(score):
    """把入流分数规整成有限实数。"""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError("score 必须是实数")
    value = float(score)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("score 必须是有限实数")
    return value


def _strength(entry):
    """强弱权重：分数高者强，同分时 seq 小（先到）者强。"""
    return (entry.score, -entry.seq)


def _weaker(first, second):
    """first 是否比 second 弱。"""
    return _strength(first) < _strength(second)


def _beats(score, seq, worst):
    """新候选是否严格强于当前最弱的在位候选；同分时在位者胜出。"""
    return score >= worst.score


class Entry:
    """一条候选记录。"""

    __slots__ = ("key", "score", "seq", "deleted")

    def __init__(self, key, score, seq):
        self.key = key
        self.score = score
        self.seq = seq
        self.deleted = False

    def __repr__(self):
        return "Entry(key=%r, score=%r, seq=%r, deleted=%r)" % (
            self.key, self.score, self.seq, self.deleted)


class StreamTopK:
    """有界流式 Top-K 内核。

    参数：
        k  结果条数上界，正整数。

    计数器（stats 的口径）：
        seen       收到的候选条数，重复入流也计入
        distinct   见过的不同 key 数，含被拒绝的
        accepted   入流后落在结果里的候选条数
        rejected   被拒绝的候选条数
        evicted    被更强候选替换出结果的条数
        updated    重复入流命中同一 key 的次数
        discarded  discard 生效的次数
        purged     物理回收的延迟删除标记数
    """

    def __init__(self, k):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k 必须是正整数")
        self._k = k
        self._heap = []
        self._index = {}
        self._live = 0
        self._next_seq = 0
        self._known = set()
        self._seen = 0
        self._distinct = 0
        self._accepted = 0
        self._rejected = 0
        self._evicted = 0
        self._updated = 0
        self._discarded = 0
        self._purged = 0

    # ------------------------------------------------------------------ 查询

    @property
    def k(self):
        """结果条数上界。"""
        return self._k

    def __len__(self):
        """当前在位的候选条数。"""
        return self._live

    def __contains__(self, key):
        """某个 key 是否正在结果里。"""
        entry = self._index.get(str(key))
        return entry is not None and not entry.deleted

    def stats(self):
        """各计数器的快照。"""
        return {
            "seen": self._seen,
            "distinct": self._distinct,
            "accepted": self._accepted,
            "rejected": self._rejected,
            "evicted": self._evicted,
            "updated": self._updated,
            "discarded": self._discarded,
            "purged": self._purged,
        }

    def peek_worst(self):
        """当前最弱的在位候选；结果为空时返回 None。

        它就是下一个被替换的对象：只有严格强于它的新候选才会挤掉它。
        """
        self._purge()
        if not self._heap:
            return None
        return self._heap[0]

    def top(self, n=None):
        """按强弱排序的在位候选；给了 n 时最多返回 n 条。"""
        if n is None:
            n = self._live
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n 必须是非负整数")
        live = [entry for entry in self._heap if not entry.deleted]
        live.sort(key=lambda entry: -entry.score)
        return live[:n]

    # ------------------------------------------------------------------ 写入

    def offer(self, key, score):
        """把一条候选交给内核。

        返回 True 表示它此刻在结果里（新占一席、更新已有席位或替换掉最弱候选），
        返回 False 表示被拒绝 —— 结果已满且它不比自己最弱的在位候选更强。
        """
        key = str(key)
        score = _normalize_score(score)
        if key not in self._known:
            self._known.add(key)
            self._distinct += 1
        entry = self._index.get(key)
        if entry is not None:
            self._update(entry, score)
            self._updated += 1
            self._accepted += 1
            return True
        self._seen += 1
        if self._live <= self._k:
            self._insert(key, score)
            self._accepted += 1
            return True
        worst = self._worst_live()
        if worst is not None and _beats(score, self._next_seq, worst):
            self._evict_worst()
            self._insert(key, score)
            self._accepted += 1
            return True
        self._rejected += 1
        return False

    def extend(self, pairs):
        """把一批 (key, score) 依次入流，返回被接受的条数。"""
        accepted = 0
        for key, score in pairs:
            if self.offer(key, score):
                accepted += 1
        return accepted

    def discard(self, key):
        """把一个 key 移出结果；返回是否真的移除了在位候选。

        只打延迟删除标记，堆数组里的空槽留到后续取堆顶时回收，
        因此这一步的开销与 k 无关。
        """
        entry = self._index.get(str(key))
        if entry is None or entry.deleted:
            return False
        entry.deleted = True
        self._discarded += 1
        return True

    # ------------------------------------------------------------------ 堆维护

    def _insert(self, key, score):
        """让一条新候选占一席。"""
        entry = Entry(key, score, self._next_seq)
        self._next_seq += 1
        self._heap.append(entry)
        self._index[key] = entry
        self._live += 1
        self._sift_up(len(self._heap) - 1)

    def _evict_worst(self):
        """把堆顶（最弱的在位候选）移出结果并修复堆序。"""
        worst = self._heap[0]
        last = self._heap.pop()
        if self._heap:
            self._heap[0] = last
            self._sift_down(0)
        self._index.pop(worst.key, None)
        self._live -= 1
        self._evicted += 1
        return worst

    def _worst_live(self):
        """最弱的在位候选；结果为空时返回 None。"""
        self._purge()
        if not self._heap:
            return None
        return self._heap[0]

    def _purge(self):
        """物理回收堆顶的延迟删除标记。"""
        while self._heap and self._heap[0].deleted:
            dead = self._heap[0]
            last = self._heap.pop()
            if self._heap:
                self._heap[0] = last
                self._sift_down(0)
            if self._index.get(dead.key) is dead:
                del self._index[dead.key]
            self._purged += 1

    def _update(self, entry, score):
        """同一 key 再次入流：分数按最后一次更新，并就地修复堆序。"""
        previous = entry.score
        if score == previous:
            return
        entry.score = score
        pos = self._heap.index(entry)
        if score > previous:
            self._sift_up(pos)
        else:
            self._sift_down(pos)

    def _sift_up(self, pos):
        """把 pos 处的候选往堆顶方向推，直到父节点不比它弱。"""
        heap = self._heap
        while pos > 0:
            parent = (pos - 1) // 2
            if not _weaker(heap[pos], heap[parent]):
                break
            heap[pos], heap[parent] = heap[parent], heap[pos]
            pos = parent

    def _sift_down(self, pos):
        """把 pos 处的候选往叶子方向推，直到子节点不比它弱。"""
        heap = self._heap
        size = len(heap)
        while True:
            left = 2 * pos + 1
            if left >= size:
                return
            if not _weaker(heap[left], heap[pos]):
                return
            heap[pos], heap[left] = heap[left], heap[pos]
            pos = left
