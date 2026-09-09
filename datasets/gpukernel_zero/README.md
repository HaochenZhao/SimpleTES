# gpukernel_zero

Isolated copy of `datasets/gpukernel` for an unguarded, paper-scale search.
Do not edit this tree to change the original `datasets/gpukernel` implementation.

Differences from upstream:

- Static checker is a no-op (no expert anti-hack / must-be-Triton gates).
- TriMul instruction is `trimul/trimul_zero.txt` (problem + score only; no
  Triton-only coaching and no "do not cache" line).
- Seed `custom_kernel` is a working naive PyTorch TriMul so SimpleTES can init.
- Eval server bind and `server_info.yaml` are local to this copy.
- Compiler concurrency default is 1 (H20 OOM-safe).

Official released `best_results/ai_foundations/trimul/trimul_best.py` is scored
on the same H20 + benchmark harness to give a local reference, not an H100 number.

Launch: `scripts/repro_gpukernel_zero/` (`start_kernel_server.sh`, `align_then_run.sh`).
Search budget: RPUCG, `(C,L,K)=(32,100,16)`, `N=51200`.
