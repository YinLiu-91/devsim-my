# DEVSIM cuDSS 优化 skill / playbook

配套架构设计见：`doc/cudss_architecture.md`
配套使用说明见：`doc/cudss_usage.md`

## 1. 目标口径

这份 playbook 不是介绍“怎么打开 cuDSS”，而是沉淀 **怎样在 DEVSIM 里做出有效的 cuDSS 优化**。

当前默认目标口径：

1. correctness 先于性能；
2. 主比较对象是 **CPU baseline vs 当前默认 `native+MT`**；
3. 结论以 **重复运行的中位数** 为主，而不是单次 wall time；
4. 先看分项 profile，再做优化判断。

当前推荐主 case：

- `examples/capacitance/cap2d_large`
- `CAP2D_LARGE_MESH_SCALE=0.1`

## 2. 已证实有效的手段

### 2.1 从 callback 主线切到 native C++ 主线

有效原因：

1. 避开 Python callback/ctypes 热路径；
2. 更容易直接接 cuDSS 的 MT / stream / native stats；
3. 能把 profile 与后处理问题更清楚地拆出来。

当前默认思路：

- `DEVSIM_CUDSS_BACKEND_POLICY=native`
- `DEVSIM_CUDSS_MT_MODE=1`

### 2.2 用 profile 先定位，不要先盲调 config

真正有效的定位手段是看：

- `cuDSS solver profile`
- `cuDSS solver iteration`
- `cuDSS transfer stats`

重点盯的不是只有 `linear_solve_seconds`，还包括：

- `iteration_load_dc_seconds`
- `iteration_device_update_seconds`
- `iteration_finalize_seconds`
- `iteration_clear_seconds`

### 2.3 修 `Device::Update` 的逐行 D2H，比继续盲开 cuDSS 参数更有收益

这次最关键的有效优化是：

1. 找到 `EquationHolder -> Equation::TryUpdateFromDevice -> copy_rows_to_host_double` 的逐行 `cudaMemcpy(sizeof(double))`；
2. 将默认路径改为优先 bulk-copy；
3. 保留 `rows` 作为显式回退和诊断开关。

结论：

- 这一步把 `Device::Update` 从 `14.59s` 量级压到约 `0.05s`；
- 当前默认 `native+MT` 的 solve-only speedup 由此跨过了历史 custom 基线。

### 2.4 用 repeated median 守门

大 case 的 total wall 会有抖动，不能拿单次结果下结论。

推荐规则：

1. `REPEATS=3` 起步；
2. 用中位数比较 total/solve；
3. raw runs 也要保留，防止“中位数看起来好、其实抖动很大”。

## 3. 当前不建议作为默认主线的做法

### 3.1 把 Python callback 路径当默认性能主线

保留理由：

- 它仍然是 fallback 和历史对照基线；
- 有助于对比 native 路径到底解决了什么问题。

但不推荐作为默认主线，因为：

- callback 热路径开销更高；
- 与当前默认 native+MT 的实际收益点不一致。

### 3.2 默认开启 stream / host threads / hybrid

当前结论：

- stream：当前不建议默认开；
- `HOST_NTHREADS=8`：已测比 default 更差；
- `HYBRID_MODE=1`：当前也低于默认；
- `REORDERING_ALG=1`：在大 case 上出现过 `FACTORIZATION status=2`。

也就是：**不要在没 profile 证据前，把这些开关当“默认优化”。**

### 3.3 只看总 speedup，不看分项耗时

这会把真正的问题藏掉。

这轮就是典型例子：

- MT 其实已经显著压低了 `linear_solve`；
- 真正拖后腿的是 `Device::Update`；
- 如果只看总时间，会误判为“MT 没生效”。

## 4. 推荐工作流

### Step 1：先做 correctness

```bash
bash scripts/run_cudss_compare_cap_large.sh
```

目标：

- 先确认当前主 case 的 baseline/cudss 输出一致；
- 不一致时先停在这里，不继续讨论性能。

### Step 2：再看主 case 的 perf compare

```bash
bash scripts/run_cudss_perf_cap_large.sh
```

要看：

1. total / solve speedup；
2. baseline 和 cudss 的 `load_dc / linear_solve / device_update / finalize / clear`；
3. raw runs 是否稳定。

### Step 3：必要时回到分项定位

若结果退化或不稳定，优先看：

1. `cuDSS solver profile`
2. `cuDSS solver iteration`
3. `cuDSS transfer stats`

定位顺序建议：

1. `linear_solve` 是否真的下降；
2. `device_update` 是否异常放大；
3. `load_dc` 是否成了主瓶颈；
4. 最后才看 config 调参。

### Step 4：最后跑 full regression

```bash
bash scripts/run_cudss_pytest_regression.sh
```

看的是：

- 有没有新的 hard fail；
- pass/skip/xfail 总体口径是否与当前已知状态一致。

## 5. 当前推荐入口

### 5.1 当前主 case correctness

```bash
bash scripts/run_cudss_compare_cap_large.sh
```

### 5.2 当前主 case perf compare

```bash
bash scripts/run_cudss_perf_cap_large.sh
```

可选覆盖：

```bash
DEVSIM_SO=/devsim/linux_x86_64_release/src/main/devsim_py3.so \
REPEATS=5 \
CAP2D_LARGE_MESH_SCALE=0.1 \
bash scripts/run_cudss_perf_cap_large.sh
```

### 5.3 full pytest regression

```bash
bash scripts/run_cudss_pytest_regression.sh
```

## 6. 一句话总结

在 DEVSIM 里做 cuDSS 优化时，**最有效的方法不是先堆 config，而是先用 correctness + profile 把瓶颈拆开，再把默认路径收敛到“最少变量、最少搬运、最少歧义”的 native+MT 主线。**
