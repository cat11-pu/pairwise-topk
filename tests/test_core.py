"""topk.core 的行为测试：k 上界、同分淘汰、堆序维护、延迟删除与统计口径。"""

import unittest

from topk.core import StreamTopK


class StreamTopKTest(unittest.TestCase):
    """有界流式 Top-K 内核。"""

    def test_k_bound_holds_and_weakest_is_replaced(self):
        """结果条数不得超过 k；装满后更强的候选必须挤掉最弱的那条。"""
        topk = StreamTopK(3)
        for key, score in (("a", 10), ("b", 20), ("c", 30)):
            self.assertTrue(topk.offer(key, score))
        self.assertEqual(len(topk), 3)
        self.assertTrue(topk.offer("d", 40))
        self.assertEqual(len(topk), 3)
        self.assertEqual([entry.key for entry in topk.top()], ["d", "c", "b"])
        self.assertEqual(topk.peek_worst().key, "b")
        self.assertFalse(topk.offer("e", 5))
        self.assertEqual([entry.key for entry in topk.top()], ["d", "c", "b"])
        self.assertEqual(topk.stats()["evicted"], 1)

    def test_equal_scores_keep_the_earlier_candidate(self):
        """同分的新来者不得挤掉在位候选：同分时先到者胜出。"""
        topk = StreamTopK(2)
        self.assertTrue(topk.offer("a", 10))
        self.assertTrue(topk.offer("b", 5))
        self.assertFalse(topk.offer("c", 5))
        self.assertEqual([entry.key for entry in topk.top()], ["a", "b"])
        self.assertEqual(topk.peek_worst().key, "b")
        self.assertEqual(topk.stats()["rejected"], 1)

    def test_worst_candidate_is_true_minimum_after_replacement(self):
        """替换堆顶之后必须修复堆序：最弱候选是真正的全局最小值。"""
        topk = StreamTopK(4)
        for key, score in (("a", 5), ("b", 9), ("c", 7), ("d", 11)):
            self.assertTrue(topk.offer(key, score))
        self.assertEqual(topk.peek_worst().key, "a")
        self.assertTrue(topk.offer("e", 100))
        self.assertEqual(topk.peek_worst().key, "c")
        self.assertEqual([entry.key for entry in topk.top()],
                         ["e", "d", "b", "c"])

    def test_discard_frees_a_slot_for_the_next_candidate(self):
        """延迟删除必须立刻让出容量：被删的候选不再占席位。"""
        topk = StreamTopK(3)
        for key, score in (("a", 1), ("b", 5), ("c", 9)):
            self.assertTrue(topk.offer(key, score))
        self.assertTrue(topk.discard("b"))
        self.assertEqual(len(topk), 2)
        self.assertEqual([entry.key for entry in topk.top()], ["c", "a"])
        self.assertTrue(topk.offer("d", 7))
        self.assertEqual(len(topk), 3)
        self.assertEqual([entry.key for entry in topk.top()], ["c", "d", "a"])
        self.assertTrue(topk.discard("a"))
        self.assertEqual(topk.peek_worst().key, "d")
        self.assertEqual(len(topk), 2)
        self.assertEqual([entry.key for entry in topk.top()], ["c", "d"])

    def test_discarded_key_re_enters_as_a_fresh_candidate(self):
        """被删掉的 key 再入流要按新候选处理，不能被重复更新吃掉。"""
        topk = StreamTopK(3)
        for key, score in (("a", 1), ("b", 5), ("c", 9)):
            self.assertTrue(topk.offer(key, score))
        self.assertTrue(topk.discard("b"))
        self.assertNotIn("b", topk)
        self.assertTrue(topk.offer("b", 100))
        self.assertEqual([entry.key for entry in topk.top()], ["b", "c", "a"])
        self.assertIn("b", topk)

    def test_score_update_repairs_the_heap(self):
        """同一个 key 改分数之后，堆顶必须换成新的最弱候选。"""
        topk = StreamTopK(3)
        for key, score in (("a", 1), ("b", 5), ("c", 9)):
            self.assertTrue(topk.offer(key, score))
        self.assertEqual(topk.peek_worst().key, "a")
        self.assertTrue(topk.offer("a", 100))
        self.assertEqual(topk.peek_worst().key, "b")
        self.assertEqual(topk.peek_worst().score, 5)

    def test_output_orders_equal_scores_by_arrival(self):
        """输出里同分候选按先到先出排列，不随堆内布局漂移。"""
        topk = StreamTopK(4)
        for key in ("a", "b", "c", "d"):
            self.assertTrue(topk.offer(key, 5))
        self.assertEqual([entry.key for entry in topk.top()],
                         ["a", "b", "c", "d"])
        second = StreamTopK(3)
        for key, score in (("a", 2), ("b", 2), ("c", 5)):
            self.assertTrue(second.offer(key, score))
        self.assertEqual([entry.key for entry in second.top()], ["c", "a", "b"])

    def test_counters_track_seen_distinct_and_updates(self):
        """统计口径：seen 含重复入流，distinct 去重，updated 记命中次数。"""
        topk = StreamTopK(8)
        for key, score in (("a", 1), ("b", 2), ("a", 3), ("c", 4)):
            self.assertTrue(topk.offer(key, score))
        stats = topk.stats()
        self.assertEqual(stats["seen"], 4)
        self.assertEqual(stats["distinct"], 3)
        self.assertEqual(stats["updated"], 1)
        self.assertEqual(stats["rejected"], 0)
        self.assertEqual([entry.key for entry in topk.top()], ["c", "a", "b"])

    def test_stream_below_capacity_keeps_every_candidate(self):
        """k 大于入流的不同 key 数时按实际条数返回，不丢也不补空位。"""
        topk = StreamTopK(8)
        self.assertEqual(topk.extend([("a", 3), ("b", 7), ("c", 5)]), 3)
        self.assertEqual(len(topk), 3)
        self.assertEqual([entry.key for entry in topk.top()], ["b", "c", "a"])
        self.assertEqual([entry.key for entry in topk.top(2)], ["b", "c"])
        self.assertEqual(topk.peek_worst().key, "a")
        self.assertIn("a", topk)
        stats = topk.stats()
        self.assertEqual(stats["accepted"], 3)
        self.assertEqual(stats["rejected"], 0)
        self.assertEqual(stats["purged"], 0)

    def test_invalid_arguments_and_empty_state(self):
        """参数校验与空结果：非法 k 或 n 报错，空内核给出空结果。"""
        with self.assertRaises(ValueError):
            StreamTopK(0)
        with self.assertRaises(ValueError):
            StreamTopK(2.5)
        empty = StreamTopK(2)
        self.assertEqual(len(empty), 0)
        self.assertIsNone(empty.peek_worst())
        self.assertEqual(empty.top(), [])
        self.assertFalse(empty.discard("missing"))
        self.assertNotIn("missing", empty)
        with self.assertRaises(TypeError):
            empty.offer("a", "high")
        with self.assertRaises(ValueError):
            empty.top(-1)


if __name__ == "__main__":
    unittest.main()
