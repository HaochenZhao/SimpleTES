# gpukernel_zero 完整运行记录

实例：`checkpoints/gpukernel_zero_trimul/2026-09-08/instance-bd60257b`  
最终 checkpoint：`db_state_144940`（2026-09-11 14:49）  
结束：`max_generations=51200` 用尽，`Evolution Complete`

## 实验设定

去护栏对照：静态检查 no-op；instruction 只保留问题和评分；种子是可跑的 naive PyTorch。  
搜索：RPUCG，`(C,L,K)=(32,100,16)`，`N=51200`，本地 gpt-oss-120b。  
分数：`combined_score = 1 / geomean(latency_ms)`，正确性 `rtol=atol=2e-2`。  
本地对照是同一套 H20 harness 上的 official `trimul_best.py`，不是论文 H100 数字。

## H20 对齐

| 程序 | score | geomean latency |
|------|------:|----------------:|
| official `trimul_best.py` | **0.48256** | **2.072 ms** |
| naive PyTorch seed | 0.06235–0.06264 | ~16.0 ms |
| 搜索 best | **0.425546** | **~2.35 ms** |
| 有护栏短跑 `(8,8,4)=256` | 0.11473 | ~8.72 ms |

相对 seed 约 **6.8×**，相对 H20 official best 约 **0.88×**。没有越过 0.48。

## 预算与吞吐

| 项 | 数字 |
|---|---|
| 生成 | **51200 / 51200**（`gen_id_counter=3200`） |
| 完成评测 | **48752** |
| 生成/评测失败 | 0 / 0 |
| 评测少于 51200 | 中间配额中断，已生成未评完的候选 resume 时没接上 |

| 阶段 | 评测并发 | 吞吐 |
|---|---:|---:|
| 前段 1 LLM 节点评测 | 8 | ~9 eval/min |
| 切到 3 评测节点 4+4 | 24 | ~18–26 eval/min |
| 最后一次续跑 12:12–14:49 | 24 | ~17 eval/min |

评测始终是瓶颈：生成一轮 512 候选大约几分钟，评测消化大约 25–60 分钟。

## 分数轨迹

| 时间 | evals | best | 说明 |
|------|------:|-----:|------|
| 09-08 17:51 | 1 | 0.06235 | naive seed |
| 09-08 18:50 | ~600 | 0.1208 | 早期 PyTorch 融合 |
| 09-08 21:05 | 2048 | 0.2032 | |
| 09-08 21:24 | | **0.3021** | 第一次大跳 |
| 09-08 22:15 | 3072 | 0.3096 | |
| 09-09 02:23 | 5120 | 0.3479 | |
| 09-09 05:15 | | 0.3488 | 随后平台到 8192 |
| 09-09 16:55 | 9216 | 0.3488 | 切 8+24 |
| 09-09 18:36 | 12096 | 0.3869 | **chain 17** 接手 |
| 09-09 23:06 | ~2.1 万 | 0.4199 | |
| 09-10 03:29 | ~26128 | **0.425546** | 最终 best，节点 `bf59d40a` |
| 09-10 03:29 之后 | →48752 | 0.425546 | 约 **2.3 万次评测零增益** |

后半段预算几乎全在同一条 `torch.compile` 捷径附近微调。

## 当前 best 在做什么

`db_state_144940/best_program.py`：

- `torch.compile(fullgraph=True)` 包整个 `custom_kernel`
- 五路线性（left/right proj + 三路 gate）拼成一次 `F.linear`
- 三角收缩改成 batched `torch.bmm`
- FP16 autocast + TF32 / cudnn benchmark
- 没有 `@triton.jit`，没有改计时器、全局 cache、未同步 stream

这是 **compiler + 精度 + 算子融合** 的 Goodhart，不是破坏评测器。原版静态检查会拦：`torch.compile`、`F.linear` / `F.layer_norm` / `torch.bmm`、缺少 Triton JIT。原版 instruction 也写了不要用 `torch.compile`、不要 persistent cache。

## 工程中断

共享盘配额（errno 122）多次把评测和日志写死，eval worker 从 24 掉到 17。最后一次从 `db_state_084225`（46080）续到预算耗尽。内存里到过的中间进度（例如 30592、46592）有几次没落盘。
