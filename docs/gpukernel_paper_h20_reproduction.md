# 官方 gpukernel（TriMul）在 H20 上的论文对齐复现

记录日期：2026-09-21  
论文：[SIMPLETES / SimpleTES, arXiv:2604.19341](https://arxiv.org/pdf/2604.19341)  
作者补遗：[wq-will/SimpleTES#6](https://github.com/wq-will/SimpleTES/issues/6)  
任务：GPUMode TriMul（Triangle Multiplicative Update），官方 guarded 评测栈  
硬件：本机 5×8 H20（96 GB），不是论文的 H100 / H200

**结论（截至写文档时）**

- 第一轮 `N=51200` 已跑完。从 naive PyTorch（~15.89 ms）搜到 **2.796 ms / score 0.3577**。
- 搜索协议对齐了论文的 kernel 特例（`C=32, K=8, L=200`）和 best-solution restart 流程；**H20 数值没有对齐**发布版 `trimul_best.py`（同 harness 上 **2.072 ms / 0.4826**），第一轮结束时慢 **1.35×**。
- 论文报的 H100 1.122 ms / H200 1.020 ms 不是这块卡上的目标。H200 的 1.020 ms 是 **两轮 × 51200** 的结果。
- 第二轮（best-solution restart）已于 2026-09-21 14:48 挂上，写文档时已到 ~2.615 ms（score 0.3824），仍在跑。

不要和下面两条线混在一起：

| 线 | 目录 / checkpoint | 是不是这次 |
|---|---|---|
| **本次：官方护栏 + 论文 kernel 特例** | `datasets/gpukernel` + `scripts/repro_gpukernel_paper` + `checkpoints/gpukernel_trimul_paper` | 是 |
| 去护栏对照 | `datasets/gpukernel_zero` + `docs/gpukernel_zero_experiment.md` | 否 |
| 早期不对齐短跑（`K=16` 等） | `scripts/repro_gpukernel_trimul` / `checkpoints/gpukernel_trimul` | 否 |

---

## 1. 论文里 GPU kernel 实际怎么跑

正文默认是 \((C,L,K)=(32,100,16)\)、\(N=51200\)。**§3.2 对所有 kernel 任务单独改过**：把 local sample 降到 \(K=8\)、仍用 \(C=32\)，以便在固定预算下多走 refinement。因此

\[
N = C \times K \times L = 32 \times 8 \times 200 = 51200,\qquad L=200
\]

仓库里 `--max-generations` 计的是候选次数，不是 prompt 数。

其它与论文 / 作者确认一致的项：

| 项 | 论文 / issue #6 | 本次 |
|---|---|---|
| 种子 | naive PyTorch（可跑） | `datasets/gpukernel/trimul/init_program_naive.py` |
| Instruction | `trimul.triton` | 同左 |
| 评测 | 官方 GPUMode 代码，`rtol=atol=2e-2`，leaderboard 模式 | 同左（`datasets/gpukernel/server`） |
| 分数 | \(1/\mathrm{geomean}(\mathrm{latency\_ms})\) | 同左 |
| 选择器 | RPUCG，\(\lambda=1.0\)，\(\gamma=0.8\) | `--selector rpucg --puct-c 1.0 --rpucg-gamma 0.8` |
| 灵感 / 失败模式 / reflection | 开；`insp` 通常 3 或 5 | `--num-inspirations 5` |
| 模型 | gpt-oss-120b | 本地 MXFP4 |
| 解码 | reasoning=high，temp=1.0，`vllm_token_forcing` | 同左 |
| 默认窗口 | context 49152 = 程序 15536 + prompt/reason 33616 | 开跑时同左；代码涨长后改 65536（见 §3.3） |
| 第二轮 | 第一轮 best 当新 \(y_0\)，**清历史**，再 51200 | 已启动，instance `d3ff8a91` |

作者在 issue #6 写明 TriMul 的 H200 数字怎么来的：

1. naive → 第一轮 51200 → **1.107 ms**
2. 该程序当新种子 → 再 51200 → **1.020 ms**（Table 5）

发布的 `best_results/ai_foundations/trimul/trimul_best.py` 是 H200 上搜到的，**没有在 H20 / MI300 上重搜**；跨卡只做直接评测。因此本机对齐目标是：**同一份发布核在本套 H20 harness 上的分数**，不是把 H100/H200 墙钟抄过来。

---

## 2. 本机对照分数（同一套 H20 + GPUMode harness）

| 程序 | combined_score | geomean latency |
|---|---:|---:|
| naive PyTorch 种子 | 0.06295 | 15.89 ms |
| 旧的不对齐 `K=16` 短跑 best | 0.181 | 5.53 ms |
| **第一轮论文对齐 best**（`48644250`，chain 22） | **0.3577** | **2.796 ms** |
| 第二轮写文档时的 best（chain 13，仍在跑） | 0.3824 | 2.615 ms |
| 发布版 `trimul_best.py` @ H20 | **0.4826** | **2.072 ms** |
| 论文 H100 / H200（参考，不是本机目标） | — | 1.122 / 1.020 ms |

去护栏的 zero 线在另一套实验里到过 ~2.35 ms（`torch.compile` + FP16），和本次官方护栏线不是同一评测栈，分数不要横比。

---

## 3. 集群与必须改的工程点

脚本根目录：`scripts/repro_gpukernel_paper/`。最终拓扑（5 节点全用）：

| 角色 | IP | 配置 |
|---|---|---|
| vLLM | `28.33.242.75`、`28.33.241.189`、`28.33.240.27` | 每节点 4×TP=2，`max-model-len=65536`，nginx 12 backend |
| kernel eval | `28.33.242.50`、`28.33.245.72` | 官方 `launch_server.py`，compile/eval GPU **4+4** |
| 调度 | head 上 `main.py` | `--gen-concurrency 48 --eval-concurrency 16` |

开跑时曾用过 1 LLM + 3 eval（生成绑死，~7–11 eval/min），再改 3 LLM + 2 eval。kernel 节点不要兼跑 vLLM。

### 3.1 种子过不了官方静态检查：`ALLOW_TORCH`

官方 `kernel_static_checker.py` 会拦 `torch.bmm` / `F.linear` / `F.layer_norm`、以及没有 `@triton.jit` 的实现。论文种子是 naive PyTorch，发布核自己也用了 `torch.bmm`。

不放开时：naive 被打成 `error` → SimpleTES `score=-inf` → RPUCG 空链 → 0 生成。空 stub 则 score=0 空转。

处理：评测服带 `SIMPLES_KERNEL_ALLOW_TORCH_IMPL=1`（`start_kernels_split.sh`）。**反作弊门仍在**（timing patch、线程、lazy tensor 等），只是允许 torch 计算算子和缺 Triton 的合法种子计分。`server_info.yaml` 由 `write_server_info.sh` 写成上述两台 kernel。

### 3.2 第一份 instance 作废

`instance-1c37f11d` 在 checker 未放行时卡在 Initial score `-inf`，已杀掉。正式第一轮是 **`instance-7bf43cb3`**。

不要用 `pkill -f gpukernel_trimul_paper`：会误匹配包装 shell，把正在干活的进程（以及操作者自己）杀掉。停搜只杀已知 PID。

### 3.3 Prompt too long → 65536 / TP=2

ConstructQuery 会把 5 份 inspiration 全文 + `trimul.triton`（约 3594 token）塞进 prompt。论文**不截断程序正文**，只截 error/traceback。代码涨到 ~30k token 后，phase-1 额度 `33616` 被吃光，客户端直接 `Prompt too long`。

论文在 ZNAA 上的同类处理是把窗口加到 65536，而不是改拼 prompt。H20 上 49152@TP=1 已经接近 91/98 GB，65536 必须 **TP=2**（每节点 4 replica）。`scripts/repro_circle_packing_26` 里的 `restart_node_vllm.sh` 写死 32768，**不能原样拿来**。

切窗口：`cutover_tp2_65536.sh`（2026-09-18 13:56）。之后第一段 44 h 里 Prompt too long 从成百上千降到 32 条；第一轮末程序再涨长，又出现 152 条（`gen_fail` 832→984）。

### 3.4 搜索进程两次静默退出

服务栈（vLLM + kernel）都还在，python 没有 traceback，日志停在 `finished_nodes=...`，偶发 `Unclosed client session`。SIGINT 能走到 `_finalize_run` 写 ckpt；自己挂掉时 **不会** finalize。

| 死亡 | 日志停 | 最后落盘 | 未写入的 eval |
|---|---|---|---|
| 2026-09-18 08:50 | 11632 | `db_state_082847`（11264） | ~368 |
| 2026-09-20 10:02 | 37840 | `db_state_075228`（36864） | ~976 |

两次都是从最新 `db_state_*` `--resume` 接上（同一棵树，不是 restart）。checkpoint manager 只留最新 `db_state`，`metadata.chain_best_scores` 可能仍停在种子 0.063，**节点库里的 score 才是真的**。

本机 torch 栈 Triton **3.5.1**，论文报 3.4.0 / 3.6.0。未强行换版本。

---

## 4. 第一轮过程（`instance-7bf43cb3`）

- 开始：2026-09-17 21:52  
- 结束：2026-09-21 11:54，`Evolution Complete`  
- 落盘：`checkpoints/gpukernel_trimul_paper/2026-09-17/instance-7bf43cb3/db_state_115425/`

| 项 | 值 |
|---|---|
| `generation_attempts` | 51200 / 51200 |
| 完成 eval | 48409 |
| gen fail / cancel | 984 / 93 |
| 有限分 / −inf | 29743 (61%) / 18666 (39%) |
| best | **0.357702 → 2.796 ms** |
| best node | `48644250391841f5a28b97a3d53233aa`，chain 22 |
| 墙钟 | ~86 h（含两次中断 + 一次 65536 切换） |
| 65536 段吞吐 | 平均 ~9.8 eval/min，后期 ~7/min |

日志里有 55 次 New best。后面大量是 tile / autotune 毛刺；**架构变了分数才上台阶**：

| 时间 | chain | 延迟 | 相对上一阶段改了什么 |
|---|---:|---|---|
| 9-17 21:55–22:00 | 4 / 7 | 15.89 → 7.60 ms | 丢掉 `einsum`，改 FP16 `F.layer_norm` + `F.linear` + `torch.bmm` |
| 9-17 23:42 | 25 | 7.16 → 6.25 ms | 第一次 Triton：五路 proj/gate 写成 `[B,H,N,N]`；N³ 用 Triton GEMM；尾巴仍是 `F.linear` |
| 9-18 00:20 | 25 | 5.79 → 4.18 ms | 输入 LN 和最终 LN+gate+proj 也进 Triton。三核骨架，尚无 autotune。随后在 ~4.15 ms 磨了十几个小时 |
| 9-18 16:58 – 9-19 11:30 | 31 | 4.15 → 3.29 ms | 同骨架 + `@triton.autotune`、更大 GEMM tile（发生在 65536 打开之后） |
| 9-19 15:54 | 22 | 3.29 → 3.19 ms | 拆开输入 LN；N³ 改回 **`torch.bmm`（cuBLAS）**，最接近发布解的收缩方式 |
| 9-20 01:16 | 27 | 3.17 → 2.89 ms | 五路权重 `torch.cat` + 一次 `F.linear`（论文 case study 的 concat-weight GEMM） |
| 9-21 04:46–08:51 | 22 | 2.89 → **2.80 ms** | 丢掉 `F.linear` / concat，六路 Triton + 很宽的 GEMM autotune，成为第一轮赢家 |

近优很少：≥0.35 的只有 65 个。chain 22 最终 2.796 ms；chain 27 的 concat 家族停在 2.851 ms；chain 31 的三核家族停在 3.165 ms。

第一轮 best 程序：841 行，无 torch 计算算子。路径：

`checkpoints/gpukernel_trimul_paper/2026-09-17/instance-7bf43cb3/db_state_115425/best_program.py`

---

## 5. 解法有没有复现论文思路

论文 case study 对 evolved TriMul 的概括：手写 Triton 四段——FP16 算 / FP32 累加、concat-weight 单次 GEMM、融合 LN+gate+projection、autotune + 自适应 `num_warps`。发布文件 `trimul_best.py`（476 行）的实际结构是：

1. Triton 输入 LN（FP32→FP16）
2. 一个 kernel 里五路 proj/gate（含 `out_gate`，五次 `tl.dot`）
3. **`torch.bmm`（cuBLAS）做 N³**
4. 一个 kernel 融合输出 LN + gate + 最终投影

本次第一轮**独立摸到了同一条路**（FP16/FP32、Triton LN、融合 proj+gate、N³、输出 LN+gate+proj、autotune），不是把 `trimul_best.py` 抄回来。但**最终赢家并不是那份 H20 解**：

| 论文关键点 | 发布版 H20 核 | 第一轮 best |
|---|---|---|
| 输入 LN | 单独 Triton | 单独 Triton |
| proj + gates | **五路融在一个 kernel** | 四路融合，`out_gate` 另开 kernel |
| concat-weight GEMM | case study 有写；发布文件是 5×`tl.dot` | 2.89 ms 家族用过，最终丢掉 |
| N³ | **`torch.bmm` / cuBLAS** | **手写 Triton batched GEMM** |
| 输出 LN+gate+最终投影 | **一个融合 kernel** | 拆成 LN×gate 和单独 final linear |
| H20 延迟 | **2.072 ms** | **2.796 ms（慢 1.35×）** |

中途更像论文的是 4.18–3.29 ms 的三核骨架，以及 3.19 ms 那次真的用了 `torch.bmm`。后半段往「更纯 Triton、拆得更开」走，H20 上只再抠了约 0.4 ms，没吃到发布解靠 cuBLAS N³ 和更紧融合拿到的那截。

---

## 6. 第二轮（best-solution restart，进行中）

这不是 `--resume`。论文 §2：一轮结束后取 \(\arg\max r\) 当新 \(y_0\)，**重置轨迹历史和实现簿记**，再开一次相同配置的全新 run。作者只默认多做这一轮（再 restart 经常饱和）。

| | `--resume`（中途续跑） | 第二轮 restart |
|---|---|---|
| instance | 同一个 `7bf43cb3` | **新的 `d3ff8a91`** |
| 树 / 访问计数 / 32 条 chain 历史 | 接着用 | 清掉 |
| `--init-program` | naive | 第一轮 `best_program.py` |
| 预算 | 用剩下的 attempts | **重新满额 51200** |

启动：

```bash
# 种子已拷到 datasets/gpukernel/trimul/init_program_round2_best.py
nohup bash scripts/repro_gpukernel_paper/run_simpletes_round2.sh \
  > scripts/repro_gpukernel_paper/logs/simpletes.log 2>&1 &
```

- 启动：2026-09-21 14:48  
- Initial score：0.3539（~2.825 ms；相对第一轮 2.796 ms 的评测抖动，不是 `-inf`）
- 写文档时（约 17:44）：`finished_nodes≈912`，best **0.382356 ≈ 2.615 ms**（chain 13），已经快过第一轮结束时的 2.796 ms  
- 相对发布版 H20 2.072 ms 仍约 **1.26×**
- 满预算估时：按 7–10 eval/min，大约还要数天

日志：`scripts/repro_gpukernel_paper/logs/simpletes.log`  
run.log：`checkpoints/gpukernel_trimul_paper/2026-09-21/instance-d3ff8a91/run.log`

---

## 7. 怎么复跑 / 续跑

```bash
# 评测服（两台 kernel，4 compile + 4 eval GPU）
bash scripts/repro_gpukernel_paper/start_kernels_split.sh 4
bash scripts/repro_gpukernel_paper/write_server_info.sh

# vLLM 65536 / TP=2 三节点 + nginx
# 见 remote_start_vllm.sh / start_vllm_node.sh / reload_nginx_llm.sh / cutover_tp2_65536.sh

# 第一轮（naive 种子）
bash scripts/repro_gpukernel_paper/run_simpletes.sh

# 同一棵树中断后续跑（不是 restart）
bash scripts/repro_gpukernel_paper/resume_simpletes.sh \
  checkpoints/gpukernel_trimul_paper/2026-09-17/instance-7bf43cb3

# 论文第二轮（新 instance，第一轮 best 当种子）
bash scripts/repro_gpukernel_paper/run_simpletes_round2.sh
```

`pick_latest_ckpt.py` 按 `completed_evaluations` 选 `db_state_*`，不要按目录名排序。

---

## 8. 尚未对齐 / 已知限制

1. **H20 墙钟没有对齐发布核 2.072 ms**；更没有对齐论文 H100/H200。
2. 第一轮赢家放弃了发布核的 cuBLAS N³ 和最终三段融合。
3. 第二轮按协议必须从第一轮 **argmax** 接着搜，不能改用中途更像论文的 3.19 ms `bmm` 家族当 \(y_0\)。
4. 本机 Triton 3.5.1 ≠ 论文 3.4.0 / 3.6.0。
5. `ALLOW_TORCH` 是为了让 naive 种子和 `F.linear`/`torch.bmm` 中间态能计分；反作弊门仍开。若关掉，种子会再次 `-inf`。
6. 搜索进程两次无 traceback 退出，原因未查清。
7. 程序过长时 65536 仍会 `Prompt too long`；论文对 TriMul 没有另写截断策略。
8. 论文还包含 batched cumsum、asymmetric matmul，本次只做了 TriMul。
