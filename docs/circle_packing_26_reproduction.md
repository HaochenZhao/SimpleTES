# Circle Packing n=26：论文问题定义下的流程、开销与复现结果

记录日期：2026-09-07  
论文：[SIMPLETES / SimpleTES, arXiv:2604.19341](https://arxiv.org/pdf/2604.19341)  
任务：单位正方形内装填 26 个互不重叠圆，最大化半径之和  
实例：`checkpoints/circle_packing_26_repro/2026-09-07/instance-a63bc1e3`  
结论：在缩减预算 `(C, L, K) = (16, 25, 8)` 下，本地 gpt-oss-120b 达到论文记录 **2.635983**，early-stop 提前结束。

本文按论文的问题定义写清：搜索在优化什么、SIMPLETES 怎么跑、这次实际花了多少查询 / 墙钟 / GPU，以及和论文默认预算差在哪里。

---

## 1. 论文中的问题定义

SIMPLETES 把每个任务写成一对 `(x0, V)`：

- `x0`：自然语言指令（本题见 `datasets/circle_packing/circle_packing_26/circle_packing_26.txt`）
- `V`：可执行评测器（本题见 `datasets/circle_packing/circle_packing_26/evaluator.py`）

目标是搜索一段程序 `s`，使标量分数 `r = V(s)` 最大。程序不直接等于几何解：模型改的是 `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END` 之间的构造代码，`run_code()` 必须返回形状 `(26, 3)` 的数组 `(x_i, y_i, r_i)`。评测器**不信任**程序自报的 Σr，而是重新做几何校验后再求和。

### 1.1 几何问题（论文 Problem O.4 / Circle Packing）

决策变量：26 个圆的圆心与半径

\[
(x_i, y_i) \in [0,1]^2,\quad r_i \ge 0,\qquad i=1,\ldots,26
\]

可行性约束（评测容差 `1e-12`）：

1. 圆必须完全落在单位正方形内  
   \(x_i - r_i \ge 0,\quad x_i + r_i \le 1,\quad y_i - r_i \ge 0,\quad y_i + r_i \le 1\)
2. 任意两圆不重叠  
   \(\|(x_i,y_i)-(x_j,y_j)\| \ge r_i + r_j\)

目标：

\[
\max \sum_{i=1}^{26} r_i
\]

任一约束不满足，或数组形状 / NaN / 负半径非法，则 `V(s) = 0`。合法解的分数就是半径之和。

### 1.2 种子与已知记录

| 来源 | Σr | 几何合法 |
| --- | ---: | --- |
| 种子网格 `init_program.py` | 1.390317 | 是 |
| 本次搜索最优程序（独立复评） | **2.6359830849176027** | 是 |
| 论文 / AlphaEvolve V2 | 2.635983 | — |
| 官方发布的 construction JSON | 2.6359830849210715 | 是 |

论文 Table 1 / Supplementary Table 18 写的是 gpt-oss-20b 与 gpt-oss-120b 都打到 **2.635983**，并认为该上界容易达到。本次复现与该数字在显示精度上一致（差约 `3.5e-12`，来自浮点与优化路径，不是另一条更优构造）。

官方发布的 `*_best.py` 依赖 `cvxpy`，主环境未装，因此没有重跑那份程序；改为用官方几何校验器直接打发布的 construction 数组。

---

## 2. SIMPLETES 搜索流程（对应论文 Algorithm 1）

论文用设计元组 `(C, L, K, Φ)` 描述一次 run：

| 符号 | 含义 | 本次取值 |
| --- | --- | ---: |
| `C` | 并行独立轨迹（chain）数 | 16 |
| `L` | 每条轨迹的精炼步数 | 25（预算） |
| `K` | 每步 local best-of-K | 8 |
| `Φ` | 父节点 / 灵感 / prompt 选择策略 | **RPUCG** |
| `N` | 评测器查询预算 | \(N = C \times L \times K = 3200\) |

每一步（论文语义）是：

1. `Φ` 在该 chain 的档案上选父节点，并采 `num_inspirations=5` 条灵感、失败模式、以及上一步赢家的 reflection。
2. LLM 生成 `K` 个候选程序。
3. 评测器 `V` 对每个候选打分（非法为 0）。
4. 该步只提交 local best；RPUCG 再对赢家写两行 `Approach` / `Insight` reflection，写入 DAG。
5. 全局最优达到 `early_stop_score` 或用尽 `N` 后停止，返回全局最优程序。

本仓库的计数方式：`--max-generations` 计的是**候选次数**（每调度一个 prompt 就 `generation_attempts += K`），因此 `max_generations=3200` 就是论文的 `N`。初始化面板打印的 `prompts/chain=25` 就是 `L`。

### 2.1 Φ = RPUCG（本次实际用的选择器）

论文默认选择器是 **RPUCG**，不是 CLI 默认的 `balance`。RPUCG 的要点：

- 在程序 DAG 上做带衰减的价值回传：\(V(s)=\max(\text{raw}(s),\;\gamma\cdot\max_{\text{child}} V(c))\)，本次 \(\gamma=0.8\)，PUCT 探索系数 \(c=0.5\)。
- `Q` / `P` 都用**全种群百分位秩**，避免不同 chain 分数尺度不一致。
- 选多个父节点时排除已选节点的一跳邻域，降低近亲繁殖。
- 每步附加 5 条 inspiration、failure-pattern 直方图、以及对 batch 赢家的短 reflection。

### 2.2 运行时（实现层，不是论文符号）

| 项 | 值 | 作用 |
| --- | --- | --- |
| gen / eval workers | 64 / 64 | 生成与评测异步并行 |
| `stream_k` | ON | 一步的 K 个候选可并行生成 |
| backpressure | 0 | 每条 chain 同时只挂一个 batch |
| init eval | 16 次取 max | 种子分数稳定化（计数上只记 1 次 `completed_evaluations`） |
| 评测超时 | 指令写 530s；引擎 `eval_timeout=3000s` | 本次最优程序复评约 518s，卡在 530s 限额边缘 |
| LLM | `openai/gpt-oss-120b` via LiteLLM | `temp=0.7`，`max_tokens=32768`，`reasoning=medium`，超时 3000s |
| `save_llm_io` | OFF | **没有逐请求 token 账单** |

---

## 3. 本次实例化与实际走了多远

### 3.1 启动命令（语义）

```text
uv run python main.py \
  --init-program datasets/circle_packing/circle_packing_26/init_program.py \
  --evaluator    datasets/circle_packing/circle_packing_26/evaluator.py \
  --instruction  datasets/circle_packing/circle_packing_26/circle_packing_26.txt \
  --model openai/gpt-oss-120b \
  --api-base http://127.0.0.1:8000/v1 --api-key EMPTY \
  --selector rpucg --num-inspirations 5 \
  --num-chains 16 --k-candidates 8 --max-generations 3200 \
  --early-stop-score 2.63598 \
  --gen-concurrency 64 --eval-concurrency 64 \
  --reasoning-effort medium \
  --output-path checkpoints/circle_packing_26_repro
```

### 3.2 引擎计数（checkpoint `metadata.json`）

| 计数器 | 值 | 解释 |
| --- | ---: | --- |
| `max_generations`（预算 N） | 3200 | \(16 \times 25 \times 8\) |
| `generation_attempts` | 1024 | 已调度的候选数；`gen_id_counter=128` × `K=8` |
| `generation_cancellations` | 8 | early-stop 取消的进行中生成 |
| `generation_failures` | 0 | 无 LLM / 解析失败 |
| `evaluation_failures` | 0 | 无评测崩溃 |
| `completed_evaluations` | 942 | 1 次种子 + 941 次候选评测完成 |
| `best_score` | 2.6359830849176027 | 节点 `a36498c0…`，chain 10 |

换算成论文符号：

- 预算：`L=25` 步 / chain，合计 400 个 prompt、3200 次 `V`。
- 实际调度：128 个 prompt ≈ 每条 chain 约 **8** 步（`128 / 16`），远小于 25。
- 实际完成的候选评测：941 次，约为预算 N 的 **29%**，约为论文默认 `N=51200` 的 **1.8%**。
- 差额 `1024 − 8 − 941 ≈ 75`：early-stop 时仍在 eval 队列 / 飞行中的候选，未写入 `completed_evaluations`。
- 另外，种子被重复评了 16 次（取 max），这 16 次不全部计入 `completed_evaluations`。

因此：这是一次**缩减预算 + 命中已知记录后提前停**的复现，不是论文默认 `N=51200` 的完整重跑。

---

## 4. 时间线与分数轨迹

日期：2026-09-07（集群本地时间）。搜索墙钟约 **42 分钟**（15:58:27 → 16:40:31）。

| 时刻 | 事件 | 分数 |
| --- | --- | ---: |
| 15:58:27 | 种子评测完成，调度开始 | 1.390317 |
| 15:58:27–16:01:22 | 第一波 64 个 gen worker 占满，eval 尚未回 | — |
| 16:02:07 | chain 5 首次小幅提升 | 1.444089 |
| 16:02:08 | chain 0 跳出局部网格，进入 2.4 区间 | 2.438966 |
| 16:07:08 | chain 5 | 2.589318 |
| 16:07:27 | chain 4 | 2.605821 |
| 16:09:30 | chain 10 | 2.611678 |
| 16:11:56 | chain 4 | 2.617835 |
| 16:14:25 | chain 11 | 2.622143 |
| 16:16:31 | chain 4 | 2.623489 |
| 16:17:30 | chain 4 | 2.623914 |
| 16:22:16 | chain 13 | 2.624165 |
| 16:24:48 | chain 5 | 2.626678 |
| 16:29:02 | chain 5 | 2.634292 |
| 16:33:51 | chain 5（同分更新节点） | 2.634292 |
| 16:40:31 | **chain 10 命中记录，early-stop** | **2.635983** |

曲线图：`scripts/repro_circle_packing_26/results/score_curve.png`。

模式：第一波生成约 3–4 分钟；一旦出现连续优化器（basinhopping + SLSQP）风格的程序，分数从 1.39 跳到 2.44，之后是约 38 分钟的精细抬升，直到贴上已知上界。

---

## 5. 开销

开销按论文口径应先报 **评测器查询数 N**，再报墙钟和加速器。本次没有打开 `save_llm_io`，因此**没有** prompt / completion / reasoning token 总量。

### 5.1 与论文默认预算对照

| 设置 | `(C, L, K)` | \(N = C L K\) | 备注 |
| --- | --- | ---: | --- |
| 论文默认（正文） | (32, 100, 16) | 51200 | gpt-oss 主实验规模 |
| 论文附录墙钟对照（vs OpenEvolve） | (16, 16, 50) | 12800 | 另一组公平比墙钟的设置 |
| **本次预算** | **(16, 25, 8)** | **3200** | 默认 N 的 6.25% |
| **本次实际用掉** | ≈ (16, 8, 8) | **≈942 完成评测** | 命中记录后停 |

论文认为 n=26 的已知上界容易达到。本次用大约 **2% 的默认查询量**就打到同一分数，与该判断一致，但不能外推到更难的 `n` 或其它任务。

### 5.2 LLM 调用量（近似）

| 类型 | 估计次数 | 说明 |
| --- | ---: | --- |
| 候选生成 | 1024 调度 / 8 取消 | 每个候选一次补全；`stream_k` 下同一步的 K 个请求并行 |
| Reflection | ≈ 已 finalize 的 batch 数（≤128） | 只对 batch 赢家写两行 Approach/Insight |
| 种子 | 0 次 LLM | 只跑 `init_program.py` |

单次生成上限 `max_tokens=32768`、`reasoning=medium`。实际输出远短于上限，但无法从 checkpoint 反推 token。若以后要报美元 / token 成本，需要 `--save-llm-io` 或在 vLLM 侧拉 `usage`。

### 5.3 评测器墙钟

- 每个候选：子进程执行 `run_code()` + `validate_solution` + `sum(r)`。
- 指令时限 530s。种子网格几乎瞬间完成；后期最优程序在独立复评中花了 **518.1s**。
- 64 路 eval 并发时，长程序会把评测侧打满，这是后半段墙钟的主因之一（不只是 LLM）。

### 5.4 推理集群与 GPU 小时

服务形态（不要用 4 节点 TP=32；gpt-oss 在本仓库里应按单卡 replica）：

| 项 | 值 |
| --- | --- |
| 硬件 | 4 节点 × 8 × H20 96GB |
| 副本 | 32 × TP=1，每节点端口 `8100–8107` |
| 汇聚 | launcher `127.0.0.1:8000` nginx `least_conn` → `/v1` |
| 权重 | 本地 gpt-oss-120b MXFP4 snapshot（约 90GB/GPU，`max_model_len=32768`） |
| 搜索墙钟 | ≈ 42 min |
| 搜索 GPU·小时 | \(32 \times 42/60 \approx 22.4\) |

22.4 GPU·小时只覆盖 **SimpleTES 搜索段**。集群上为避免空载被杀，vLLM 与 keepalive 在搜索前后还继续占卡，那部分不算进论文 N，但算进集群占用。

Serving 脚本：`scripts/repro_circle_packing_26/`。

---

## 6. 搜到的程序在做什么

最优程序：`scripts/repro_circle_packing_26/results/best_program.py`  
对应节点：`a36498c0e0cd41ee9f20fca97f04f027`

它不再手摆圆心，而是把论文约束写成数值优化：

1. 变量 \(v = (x_{1:n}, y_{1:n}, r_{1:n})\)，目标 \(\min -\sum r_i\)。
2. SLSQP 不等式：边界 \(x_i-r_i\ge 0\) 等，以及 \((x_i-x_j)^2+(y_i-y_j)^2 \ge (r_i+r_j)^2\)。
3. 起点池：贪心 max-min 放置 + 扰动 + 随机可行点。
4. 对每个起点做 time-bounded `basinhopping`（内层仍是 SLSQP），总预算约 470s，给评测器 530s 留余量。

也就是说，SIMPLETES 搜到的是**带约束的全局/局部优化器**，不是一张固定圆心表。分数由评测器对返回数组重新验证，因此「程序自报 Σr」不能当作论文分数。

---

## 7. 为何算作按论文标准成功

1. **同一任务、同一目标**：单位正方形 n=26，最大化 Σr，非法为 0。
2. **同一选择器族**：RPUCG，而不是仓库 CLI 默认的 `balance`。
3. **同一模型族**：gpt-oss-120b（本地 vLLM OpenAI API，不是闭源 API）。
4. **分数对齐**：独立复评 2.63598308，与论文 2.635983 / 官方 construction 2.63598308 一致且几何合法。
5. **预算更小**：用 3200 的上限、实际约 942 次评测打到记录，支持论文「该实例容易碰到上界」的说法。

不能声称的部分：

- 没有复现论文默认 `N=51200` 的完整曲线或方差。
- 没有 20b 对照，也没有和 OpenEvolve 的同预算墙钟赛。
- 没有 token / 美元成本。
- 没有重跑官方 `cvxpy` 版 `*_best.py`。

---

## 8. 工程上影响复现的点

这些不改变问题定义，但决定这次能不能跑完、GPU 会不会被空载杀掉。

1. **占卡**：H20 集群空载会被杀。先 `burn_node.py`，vLLM 起来后再 keepalive；停 burn 必须用 pid，不能 `pkill -f burn_node.py`（会杀掉启动脚本自己）。
2. **并行方式**：32 路 TP=1 replica + nginx，不要复用 `hy4_serve/` 的 4 节点 TP=32。
3. **Harmony vocab**：`HF_HUB_OFFLINE=1` 时 `openai_harmony` 需要 `/tmp/tiktoken-rs-cache/` 下的 vocab。后两台机器因此挂掉 16 路 replica；拷到共享盘并同步到 worker 后 32/32 的 `/v1/models` 才 200。
4. **代理**：集群 `no_proxy` 不含 worker IP 时，访问 `244.19` / `243.17` 会进 Squid 503。探测用 `curl --noproxy '*'`。
5. **vLLM 0.13.0** 没有 `--async-scheduling`，不要传。
6. **Keepalive 力度**：短请求看起来 200，但 GPU util 仍接近 0。需要 16–32 worker、`max_tokens` 128–256 量级才能稳住利用率。

---

## 9. 产物路径

| 内容 | 路径 |
| --- | --- |
| 本说明 | `docs/circle_packing_26_reproduction.md` |
| 短结果表 | `scripts/repro_circle_packing_26/results/REPORT.md` |
| 独立复评 JSON | `scripts/repro_circle_packing_26/results/verify_report.json` |
| 最优程序副本 | `scripts/repro_circle_packing_26/results/best_program.py` |
| 搜索曲线 | `scripts/repro_circle_packing_26/results/score_curve.png` |
| Run 日志 | `checkpoints/circle_packing_26_repro/2026-09-07/instance-a63bc1e3/run.log` |
| Checkpoint | `.../instance-a63bc1e3/db_state_164037/` |
| 集群脚本 | `scripts/repro_circle_packing_26/` |

停服务（若仍在占卡）：`scripts/repro_circle_packing_26/stop_vllm.sh`，并结束 `/tmp/vllm_keepalive.pid`。
