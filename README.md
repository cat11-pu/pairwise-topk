# topk

一个纯内存、确定性的流式 Top-K 内核：有界最小堆、同分时的先到者胜出、延迟删除
标记与惰性回收，以及入流、淘汰、更新的计数口径。候选由调用方按顺序喂入，内核
不读时钟、不起线程，也不做任何 I/O。

## 目录

- topk/core.py：内核实现（有界最小堆、淘汰判定、延迟删除与统计）
- tests/test_core.py：内核的行为测试

## 怎么跑测试

在项目根目录执行：

    python3 -m unittest discover -s tests -v

Windows 上把 python3 换成你的解释器路径，例如：

    C:/Users/<你>/AppData/Local/Programs/Python/Python313/python.exe -m unittest discover -s tests -v
