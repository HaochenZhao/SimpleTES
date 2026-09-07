# Circle packing n=26 reproduction

SimpleTES + local gpt-oss-120b reached the paper record.

论文问题定义、`(C,L,K,Φ)` 流程、查询/墙钟/GPU 开销与工程注意点见 [docs/circle_packing_26_reproduction.md](../../../docs/circle_packing_26_reproduction.md)。

| Source | Score (sum of radii) | Valid |
| --- | ---: | --- |
| Seed grid (`init_program.py`) | 1.390317 | yes |
| This run (independent re-eval) | **2.6359830849176027** | yes |
| Paper / AlphaEvolve V2 | 2.635983 | — |
| Official released construction JSON | 2.6359830849210715 | yes |

- Search: `rpucg`, `(C, L, K) = (16, 25, 8)`, early-stop at 2.63598.
- Stopped after 942 evaluations (budget 3200) when score `2.63598308 >= 2.635980`.
- Generation failures: 0.
- Instance: `checkpoints/circle_packing_26_repro/2026-09-07/instance-a63bc1e3`
- Curve: `score_curve.png`

The released `*_best.py` was not re-executed (missing `cvxpy` in the main venv). The released construction array was checked with the official geometry validator instead.
