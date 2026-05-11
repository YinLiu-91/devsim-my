# DEVSIM cuDSS 使用与验证说明

配套优化方法论与复盘手册见：`doc/cudss_optimization_skill.md`

## 1. 环境准备

建议安装（Python runtime 方式）：

```bash
pip install nvidia-cudss-cu12
```

如果动态库不在默认搜索路径，可设置：

```bash
export DEVSIM_CUDSS_LIB=/path/to/libcudss.so.0
```

native C++ 后端现在也会自动扫描常见 Python runtime 安装位置，例如：

- `/usr/local/lib/python*/dist-packages/nvidia/cu*/lib/libcudss.so*`
- `/usr/local/lib/python*/site-packages/nvidia/cu*/lib/libcudss.so*`

因此像 `pip install nvidia-cudss-cu12` 这种安装方式，即使 `libcudss` 不在 linker cache，native 路径通常也能直接找到库。

调试 API 调用状态：

```bash
export DEVSIM_CUDSS_DEBUG=1
```

打印 C++/shim 侧统计（可选）：

```bash
export DEVSIM_CUDSS_PROFILE=1
```

配置 cuDSS 后端策略（推荐）：

```bash
export DEVSIM_CUDSS_BACKEND_POLICY=auto    # auto/native/callback
```

兼容旧开关（仍可用）：

```bash
export DEVSIM_CUDSS_NATIVE_CPP=1           # 等价于 BACKEND_POLICY=native
```

可选指定动态库路径：

```bash
export DEVSIM_CUDSS_LIB=/path/to/libcudss.so.0
export DEVSIM_CUDART_LIB=/path/to/libcudart.so
```

## 2. 在脚本中启用 cuDSS

```python
import devsim
# 仓库源码运行
from cudss.cudss_shim import local_solver_callback
# wheel 安装后可使用: from devsim.cudss.cudss_shim import local_solver_callback

devsim.set_parameter(name="direct_solver", value="cudss")
devsim.set_parameter(name="solver_callback", value=local_solver_callback)
# 可选：host(默认) / device_experimental
devsim.set_parameter(name="cudss_result_mode", value="host")
```

调用链速记：

- `devsim.solve(...)`
- C++ `solveCmd` 分发
- `direct_solver=cudss` 触发 `solver_callback`
- `cudss_shim` 调用 `cudss_loader`
- 进入 cuDSS C API

执行阶段速记：

- `init`：创建句柄/配置；
- `factor` + `is_same_symbolic=False`：`ANALYSIS` + `FACTORIZATION`；
- `factor` + `is_same_symbolic=True`：`REFACTORIZATION`；
- `solve`：`SOLVE`。

## 3. pytest 迁移流程与对比验证（baseline vs cuDSS）

新增 pytest 框架位于：

- `testing/pytest/conftest.py`
- `testing/pytest/case_parser.py`
- `testing/pytest/run_case.py`
- `testing/pytest/test_cudss_compare.py`

迁移/执行流程：

1. `case_parser.py` 从 `CTestTestfile.cmake` 解析 `add_test` 与 `DEPENDS`；
2. `run_case.py` 统一子进程执行 case，并按 `--solver-mode` 切换 baseline/cudss；
3. `test_cudss_compare.py` 先跑 baseline，再跑 cuDSS，并做输出对比；
4. 按依赖链优先执行前置 case，避免缺失中间产物导致误报。

### 3.1 基本命令

```bash
pytest -q testing/pytest
```

当前推荐入口（默认 `native+MT`）：

```bash
bash scripts/run_cudss_perf_cap_large.sh
bash scripts/run_cudss_compare_cap_large.sh
bash scripts/run_cudss_pytest_regression.sh
```

分别对应：

1. `run_cudss_perf_cap_large.sh`：baseline vs 当前默认 `native+MT` 的性能对比；
2. `run_cudss_compare_cap_large.sh`：当前 `cap2d_large` 的 strict correctness compare；
3. `run_cudss_pytest_regression.sh`：整套 pytest regression。

默认情况下不需要手工记住 cuDSS 相关环境变量。若要覆盖默认值，只需要在命令前设置少量参数：

- `DEVSIM_SO`：指定使用的 `devsim_py3.so`
- `CAP2D_LARGE_MESH_SCALE`：指定 `cap2d_large` 规模
- `REPEATS`：仅性能脚本使用，默认 `3`

性能脚本会直接输出：

1. baseline / cudss 的 `total(s)`、`solve(s)`；
2. `load_dc / linear_solve / device_update / finalize / clear` 分项耗时；
3. `baseline/cudss` 的 total 和 solve-only speedup；
4. 每次重复的 raw run，便于人工核对抖动。

放大电容 2D 基准规模（用于观察大规模下 cuDSS 收益）：

```bash
CAP2D_LARGE_MESH_SCALE=0.7 pytest -q -s testing/pytest \
  --solver-mode=both \
  --case-filter='examples/capacitance/cap2d_large' \
  --print-timing \
  --devsim-so=/devsim/linux_x86_64_release/src/main/devsim_py3.so
```

说明：
- `examples/capacitance/cap2d_large.py` 支持环境变量 `CAP2D_LARGE_MESH_SCALE`；
- 数值越小网格越密、方程规模越大（例如 `1.0 -> 0.7` 可明显增大 `number of equations`）。

默认行为：

1. 每个 case 先跑 baseline；
2. 再跑 cuDSS；
3. 对比 baseline/cudss 的输出一致性（带浮点容差规整）。
4. 对于当前 Phase-1 不支持（AC/Noise/Transient）的 case：`skip`。
5. 对于 cuDSS 收敛失败、或 baseline/cudss 输出不一致：默认 `xfail`（不阻断整体回归）；`--strict-cudss` 下转为 hard fail。

### 3.2 `--solver-mode` 行为说明

`--solver-mode` 是 pytest 层参数（`testing/pytest/conftest.py`），支持：

- `--solver-mode=baseline`
  - 仅执行 baseline 分支；
  - 不执行 cuDSS 分支，不做 baseline/cudss 输出对比。
- `--solver-mode=cudss`
  - 仅执行 cuDSS 分支；
  - 若无 GPU/cuDSS runtime，直接 `skip`。
- `--solver-mode=both`（默认）
  - 先跑 baseline，再跑 cuDSS；
  - 仅在 `both` 下做 baseline/cudss 输出对比与 timing 汇总。

### 3.3 `run_case.py` 如何切换求解器

`testing/pytest/run_case.py` 的 `--solver-mode` 只接受 `baseline/cudss`，由 pytest 主流程在 `both` 模式下分别调用两次。

- cuDSS 路径（`_enable_cudss_solver`）：

```python
devsim.set_parameter(name="direct_solver", value="cudss")
devsim.set_parameter(name="solver_callback", value=local_solver_callback)
```

- baseline 路径（`_enable_baseline_solver_if_needed`）：
  - 先读 `direct_solver`；
  - 仅当 `direct_solver == "unknown"` 时，才设置：

```python
devsim.set_parameter(name="direct_solver", value="custom")
devsim.set_parameter(name="solver_callback", value=umfpack_solver_callback)
```

也就是说：baseline 分支是“按需覆盖”；cuDSS 分支会显式注入 cuDSS solver callback，而实际走 `custom callback` 还是 `native cudss`，由 `DEVSIM_CUDSS_DIRECT_SOLVER` 控制。

- 直接运行 `pytest testing/pytest/test_cudss_compare.py` 时，若不额外设置环境变量，当前默认是 `DEVSIM_CUDSS_DIRECT_SOLVER=custom`；
- `bash scripts/run_cudss_pytest_regression.sh` 会额外导出 `native+MT` 默认环境，因此这是“当前默认主线”的回归口径。

### 3.4 常用参数

只跑 baseline：

```bash
pytest -q testing/pytest --solver-mode=baseline
```

只跑 cuDSS：

```bash
pytest -q testing/pytest --solver-mode=cudss
```

按正则筛选 case：

```bash
pytest -q testing/pytest --case-filter='testing/cap2|testing/mesh2d'
```

限制 case 数量（调试）：

```bash
pytest -q testing/pytest --max-cases=10
```

指定 devsim 模块（建议 debug 版本）：

```bash
pytest -q testing/pytest --devsim-so=/devsim/linux_x86_64_debug/src/main/devsim_py3.so
```

开启严格模式（cuDSS 失败即 fail）：

```bash
pytest -q testing/pytest --strict-cudss
```

打印每个 case 的 timing 汇总（仅 `both` 有意义）：

```bash
pytest -q testing/pytest --solver-mode=both --print-timing --case-filter='testing/cap2'
```

输出 JSONL timing 报表（仅 `both` 有意义）：

```bash
pytest -q testing/pytest --solver-mode=both --timing-json=/tmp/cudss_timing.jsonl --case-filter='testing/cap2'
```

当前环境一次全量执行示例：

```bash
pytest -q testing/pytest
# 60 passed, 15 skipped
```

## 4. 无 GPU/cuDSS 与 strict 模式行为

pytest 框架会自动探测 cuDSS 可用性：

- 有 GPU/cuDSS：按 `--solver-mode` 执行对应分支；
- 无 GPU/cuDSS：
  - `--solver-mode=cudss`：case 直接 `skip`；
  - `--solver-mode=both`：baseline 先执行，随后 cuDSS 分支 `skip`（该 case 最终记为 skip）。

`--strict-cudss` 行为：

- 默认（非 strict）：
  - cuDSS 执行失败 -> `xfail`
  - baseline/cudss 输出不一致 -> `xfail`
- strict（`--strict-cudss`）：
  - 上述两类情况都提升为 hard fail（`assert` 失败）。

## 5. 迭代时的 H2D / D2H 与优化方向

当前实现里：

- `Ap/Ai`：pattern 首次上传，pattern 变化时重传；
- `Ax`：每次数值分解时更新；
- `RHS`：每次 `solve` 前上传；
- `X`：`solve` 后回传到 host。

这意味着 cuDSS 主要变慢的原因通常不是单个 API，而是：

1. 小 case 中固定开销太高；
2. host/device 往返太频繁；
3. Python callback 额外增加调用边界开销。

优化顺序建议：

1. 先复用 device buffer，减少 `cudaMalloc/cudaFree`；
2. 复用符号分析，尽量让 `ANALYSIS` 只跑一次；
3. 让 `factor` 只更新 `Ax`，`solve` 只更新 `RHS`；
4. 若要进一步减少开销，继续沿当前已落地的 native C++ 路径推进，避免再把默认方案拉回 callback 主线。

### 5.1 GitHub 调研结论（Newton/非线性迭代 + cuDSS）

参考代码与可迁移模式：

1. `cvxgrp/scs`  
   - 文件：`linsys/cudss/direct/private.c`  
   - 关键模式：初始化做 `ANALYSIS + FACTORIZATION`；迭代更新做 `REFACTORIZATION`；RHS/解向量 pinned staging。
2. `ginkgo-project/ginkgo`  
   - 文件：`extensions/cuda/solver/cudss.cpp`  
   - 关键模式：`refactorize()` 明确仅做 `REFACTORIZATION`；可配 `cudssConfigSet`（reordering/hybrid）；绑定 stream。
3. `owensgroup/RXMesh`  
   - 文件：`include/rxmesh/diff/newton_solver.h`  
   - 关键模式：Newton 框架内对线性求解阶段做独立计时（`pre_solve + solve`），便于定位瓶颈。

对 DEVSIM 的映射（当前状态）：

1. 已对齐：`ANALYSIS/REFACTORIZATION` 分阶段调用、pinned staging 开关、phase 级计时输出。  
2. 进行中：analysis 复用命中率提升（结构指纹复用）、solver 包装耗时与 phase 耗时分离。  
3. 待实施：`cudssConfigSet` 可调面（reordering/hybrid）系统化接入与参数扫描。

已落地（当前实现）：

1. `cudss_shim.py` 首次 `factor` 阶段不再为 RHS/X 做零值 H2D 初始化拷贝（改为 device 侧直接分配）。
2. 结构不变时仅更新 `Ax` 数值，不再重复设置 A/B/X matrix value 指针。
3. 避免在 `factor` 后额外 `cudaDeviceSynchronize()`，保留 `solve` 路径的同步与 D2H 回传。
4. `direct_solver=cudss` 通过 `CuDSSPreconditioner` 接入，symbolic 变化时才上传 `Ap/Ai`。
5. `cudss_shim` 支持 symbolic 变化后的设备结构重建，并提供 `stats` 统计（H2D/D2H 字节与 phase 次数）。
6. 新增 `cudss_result_mode`：
   - `host`（默认）：保持现有行为；
   - `device_experimental`：开启 L2 实验通路（当前仍保持 host 回传兼容）。
7. 可选实验开关：
   - `DEVSIM_CUDSS_PINNED_STAGING=1`：默认开启，使用 pinned staging 传输 RHS/X（callback 与 native C++ 路径均可生效）；可设 `0` 关闭；
   - `DEVSIM_CUDSS_ZERO_COPY_EXPERIMENT=1`：让结果 X 走 mapped pinned host 内存，避免显式 D2H。
   - `DEVSIM_CUDSS_BACKEND_POLICY=auto|native|callback`：选择 cuDSS 后端策略（默认 `auto`）。
   - `DEVSIM_CUDSS_NATIVE_CPP=1`：兼容旧开关，等价 `DEVSIM_CUDSS_BACKEND_POLICY=native`。
   - `DEVSIM_CUDSS_MT_MODE=1`：在 native 路径尝试启用 cuDSS MT mode（analysis 阶段常见收益点）。
   - `DEVSIM_CUDSS_THREADING_LAYER=/path/to/libcudss_mtlayer_*.so`：指定 MT mode threading layer（默认 `libcudss_mtlayer_gomp.so.0`）。
   - `DEVSIM_CUDSS_SYMBOLIC_HASH_REUSE=1`：native 路径启用结构指纹复用（当 symbolic 状态保守失配时，基于 `Ap/Ai` hash 尝试复用 analysis 结果）。
   - `DEVSIM_CUDSS_REORDERING_ALG=<int>`：native 路径设置 `CUDSS_CONFIG_REORDERING_ALG`（命名友好入口）。
   - `DEVSIM_CUDSS_HYBRID_MODE=<int>`：native 路径设置 `CUDSS_CONFIG_HYBRID_MODE`。
   - `DEVSIM_CUDSS_HYBRID_EXECUTE_MODE=<int>`：native 路径设置 `CUDSS_CONFIG_HYBRID_EXECUTE_MODE`。
   - `DEVSIM_CUDSS_HOST_NTHREADS=<int>`：native 路径设置 `CUDSS_CONFIG_HOST_NTHREADS`。
   - `DEVSIM_CUDSS_CONFIG_SET='param=value,param=value'`：透传 `cudssConfigSet`（native 路径）；可与上述命名开关叠加，均计入 `config_set_applied`。
8. `DEVSIM_CUDSS_PROFILE=1` 时，`cuDSS profile` 额外输出 analysis 失配分类计数：
   - `analysis_miss_first_factor_calls`
   - `analysis_miss_symbolic_status_calls`
   - `analysis_miss_hash_mismatch_calls`
   - `analysis_miss_dim_change_calls`
   - `analysis_miss_backend_mode_calls`
9. 参数扫描入口（用于收敛 `cudssConfigSet` 组合）：
    - `python3 testing/pytest/cudss_config_scan.py --case-filter='examples/capacitance/cap2d_large'`
    - `--modes=native,native_mt,native_mt_stream` 可直接比较 native / MT / stream 三条路径
    - `--repeats=<N>` 可输出中位数结果，降低单次波动影响。
10. pytest `run_case.py` 选择 cuDSS 路径时支持：
   - `DEVSIM_CUDSS_DIRECT_SOLVER=custom|cudss`（默认 `custom`）。
   - 对不支持 `direct_solver=cudss` 的构建，建议保持 `custom` 以避免用例直接失败。
11. `cudss_shim`（custom 路径）现已支持 `cudssConfigSet`：
    - 继续支持 `DEVSIM_CUDSS_CONFIG_SET='param=value,...'` 与命名参数开关。
    - 可选启用 `DEVSIM_CUDSS_AUTO_REORDERING=1`：
      - `n <= DEVSIM_CUDSS_AUTO_REORDERING_THRESHOLD`（默认 `1024`）自动选 reordering=`1`
      - `n > threshold` 自动选 reordering=`0`
    - `DEVSIM_CUDSS_CONFIG_FALLBACK_ON_UNSUPPORTED=1`（默认开启）：当调参组合触发 `CUDSS_STATUS_NOT_SUPPORTED` 时自动回退默认 config 并重试，避免整条用例失败。
12. native C++ 路径新增可选 stream 绑定：
    - `DEVSIM_CUDSS_USE_STREAM=1` 时，若 `cudssSetStream` 与 `cudaStream*` 符号可用，则为 cuDSS 绑定独立 CUDA stream；
    - profile 输出会额外打印 `stream_mode=1/0`，用于确认是否真的命中该路径。

## 6. L2 `device_experimental` 当前进展

当前已打通：

1. `cudss_result_mode=device_experimental` 开关已贯通到 C++ `CuDSSPreconditioner` 与 `cudss_shim`。
2. `solve` 返回中包含设备结果元信息（`x_location`、`x_device_token`）。
3. Newton 迭代已消费该元信息（写入迭代信息对象），用于后续扩展真实 device 消费链。
4. 底层已引入 `ResultView` 抽象，作为后续替换 host-only 更新签名的过渡层。

当前限制：

1. 仍走 host 回传兼容路径（`x_location=device_experimental_host_fallback`）。
2. 因数学层更新链路仍以 host 向量为输入，D2H 尚未下降到 0。

本地回归观察（`cap2`，开启 `DEVSIM_CUDSS_PROFILE=1`）：

- `host` 与 `device_experimental` 模式下，`d2h_bytes` 目前一致（例如 `192` / `96`），说明功能已贯通但尚未完成“去 D2H”。

### 6.1 zero-copy 实验

`DEVSIM_CUDSS_ZERO_COPY_EXPERIMENT=1` 会把结果 `X` 放到 mapped pinned host 内存里，由 cuDSS 直接写回主机可见缓冲，减少一次显式 `cudaMemcpy(D2H)`。

该模式仍然是实验性路径，适合用来验证“去 D2H”的收益上限；它不改变上层 `ResultView`/`Device::Update` 的 host 兼容语义。

## 7. 关键模式落地状态（对齐 cvxgrp 思路）

已落地并默认生效（可通过环境变量回退）：

1. 初始化阶段执行 `ANALYSIS + FACTORIZATION`；
2. symbolic 结构不变的迭代更新执行 `REFACTORIZATION`；
3. `RHS/解向量` 传输默认使用 pinned staging（`DEVSIM_CUDSS_PINNED_STAGING=1`，可设 `0` 关闭）。

### 7.1 当前默认 `native+MT` 一键性能复现

复现脚本：`scripts/run_cudss_perf_cap_large.sh`

```bash
bash scripts/run_cudss_perf_cap_large.sh
```

若要显式指定构建与重复次数：

```bash
DEVSIM_SO=/devsim/linux_x86_64_release/src/main/devsim_py3.so \
REPEATS=3 \
CAP2D_LARGE_MESH_SCALE=0.1 \
bash scripts/run_cudss_perf_cap_large.sh
```

当前 polish 后的一轮 `REPEATS=3` 中位数结果（机器负载会带来波动，最终以脚本当次输出为准）：

| mode | total(s) | solve(s) | total speedup | solve-only speedup |
| --- | ---: | ---: | ---: | ---: |
| baseline | `95.793514` | `59.205752` | - | - |
| cudss native+MT（当前默认） | `47.482376` | `13.430366` | `2.017x` | `4.408x` |

脚本还会额外打印两边的 `load_dc / linear_solve / device_update / finalize / clear` 分项中位数，便于直接对照 CPU 与 cuDSS 的 solver framework 开销分布。

### 7.2 P4（device assembly）现状

当前完成的是“求解后结果消费链”与“RHS/X 传输链”优化；`RHS`/`Ax` 真正 device-side 组装尚未打通。当前 Newton 主循环仍通过 `LoadMatrixAndRHS(...)` 在 host 侧生成 `rhs` 向量后再传入求解器，因此完整的 device assembly 下沉仍需新增装配阶段的设备侧接口。

### 7.3 当前 correctness / pytest regression 状态

当前 `cap2d_large(scale=0.1)` 与 baseline 的 strict 对比命令：

```bash
bash scripts/run_cudss_compare_cap_large.sh
```

当前结果：

- `cap2d_large` strict compare：`1 passed`
- 全量 callback compare harness（`pytest -q testing/pytest/test_cudss_compare.py`）
  - `60 passed, 15 skipped`
  - 这条口径当前已**全部收口**：先用 `||Ax-b||∞` 守卫收掉 `ptest2` 这类明显线性残差失真 case，再用“每个 solve session 前 3 轮 second opinion”把 `mos_2d / diode / gmsh_mos2d` 这类早期病态 Jacobian 分支也切回 UMFPACK fallback。
- 全量 `bash scripts/run_cudss_pytest_regression.sh`
  - `43 passed, 15 skipped, 17 xfailed`
  - `0 failed`

结论：当前默认 `native+MT` 路径下，**当前主 case 与 baseline 对比正确，且整套 pytest 没有出现新的 hard fail**；主线口径仍是 `43 passed, 15 skipped, 17 xfailed`。同时，callback compare harness 已从 `43/15/17` 继续收口到 **`60 passed, 15 skipped`**。另外，`examples/diode/diode_2d` 这类 `solve(info=True)` 结构化诊断输出现在也已按现有迭代日志同口径过滤，不再把 solver-only 诊断差异当成 correctness mismatch。

## 8. solver-other 新基线（百万级 custom 路径）

测试口径：

- `CAP2D_LARGE_MESH_SCALE=0.1`
- `DEVSIM_CUDSS_DIRECT_SOLVER=custom`
- `DEVSIM_CUDSS_PROFILE=1`
- `pytest --solver-mode=both --print-timing --case-filter='examples/capacitance/cap2d_large'`

### 8.1 总体结果

| case | eqns | baseline_total(s) | cudss_total(s) | baseline_solver(s) | cudss_solver(s) | total speedup | solve-only speedup | Newton 迭代次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `examples/capacitance/cap2d_large` | `1373662` | `71.949466` | `48.645135` | `38.968433` | `15.092723` | `1.479x` | `2.582x` | `2` |

### 8.2 solver 总耗时分层求和

当前已经验证：

`cudss_solver_s`
`= custom_factor_total`
`+ custom_solve_total`
`+ external_factor_overhead`
`+ external_solve_overhead`
`+ linear_wrapper_other`
`+ solver_framework`
`+ solver_python_wrap`

| layer | seconds |
| --- | ---: |
| `custom_factor_total_seconds` | `2.608428` |
| `custom_solve_total_seconds` | `0.425410` |
| `external_factor_overhead_seconds` | `0.011274` |
| `external_solve_overhead_seconds` | `0.047861` |
| `linear_wrapper_other_seconds` | `0.934285` |
| `solver_framework_seconds` | `11.065920` |
| `solver_python_wrap_seconds` | `0.000000` |
| `sum` | `15.092723` |

结论：这条 `2.582x solve-only` 的大 case 里，cuDSS 内核并不是主瓶颈；最大的 “other” 实际上是 solver 框架层，尤其是 `LoadMatrixAndRHS(DC)`。

### 8.3 solver framework 逐项分布

| bucket | seconds |
| --- | ---: |
| `LoadMatrixAndRHS(DC)` | `9.184530` |
| `Finalize` | `0.867084` |
| `Device::Update` | `0.571625` |
| `ClearMatrix` | `0.161209` |
| 其余 setup/post/print/error 等 | `0.281472` |
| `solver_framework_seconds` | `11.065920` |

### 8.4 两次 Newton 迭代明细

| iter | total(s) | load_dc(s) | linear_solve(s) | device_update(s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | `12.554100` | `8.105360` | `3.201670` | `0.283995` |
| 1 | `1.691220` | `1.079170` | `0.255414` | `0.287630` |

说明：

1. 这次不是 “内部 solve 调了两次就等于 Newton 两次”；而是**顶层 solve 调用 1 次，Newton 迭代了 2 次**。
2. 当前 `cap2d_large` 是线性电容例子，不适合作为“拉高 Newton 迭代次数”的主扫描载体。

## 9. 迭代次数扫描基线（改用非线性 diode benchmark）

因为 `cap2d_large` 本身是线性问题，所以本轮新增了专用非线性 benchmark：

- case：`examples/diode/diode_1d_cudss_bench`
- 脚本：`examples/diode/diode_1d_cudss_bench.py`
- 扫描脚本：`python3 scripts/run_cudss_iteration_scan.py`

它的目标不是替代百万级 speedup 基线，而是回答另一个问题：**当最终一次 DC Newton 迭代次数增加时，baseline/cudss 的 solver speedup 如何变化**。

当前默认扫描维度不再只看 bias；更稳定的做法是：

- 固定 `bias=0`
- 通过 `DIODE_1D_BENCH_POTENTIAL_OFFSET` 人为扰动最终一次 DC 的初值
- 这样可以稳定把最终 Newton 迭代数从 `1` 拉到 `5~6`
- 同时避免大 bias 直接触发不收敛

输出字段包括：

- `baseline_newton_iterations`
- `cudss_newton_iterations`
- `speedup`
- `solve_only_speedup`
- `mt_mode`
- `stream_mode`

### 9.1 当前可复现扫描结果（custom 路径）

命令：

`python3 scripts/run_cudss_iteration_scan.py --biases=0.0 --carrier-scales=1.0 --potential-offsets=0.0,0.001,0.01,0.05,0.1 --modes=custom --repeats=1 --ramp-steps=0`

| potential_offset | baseline iter | cudss iter | baseline_solver(s) | cudss_solver(s) | solve-only speedup | total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.000` | `1` | `1` | `0.036853` | `0.569058` | `0.064761x` | `0.180768x` |
| `0.001` | `2` | `2` | `0.037846` | `0.580114` | `0.065239x` | `0.181884x` |
| `0.010` | `3` | `2` | `0.041845` | `0.594873` | `0.070343x` | `0.181675x` |
| `0.050` | `4` | `3` | `0.032034` | `0.623517` | `0.051376x` | `0.158628x` |
| `0.100` | `6` | `5` | `0.053018` | `0.675186` | `0.078523x` | `0.180060x` |

结论：

1. 这套非线性 benchmark 已经能稳定得到不同 Newton 迭代次数。
2. 但在当前 custom callback 路径下，迭代次数增加并没有把 cuDSS 拉到更有利区间；小规模非线性 case 仍主要受 callback/包装固定开销支配。
3. 因此“增加迭代次数就自然体现 cuDSS 加速比”这件事，在当前小规模 diode benchmark 上**并不成立**；真正显著的 solve-only 提升还是出现在百万级 `cap2d_large` 这类大矩阵 case。

### 9.2 native / native+MT / native+stream 状态

在当前构建和这条 diode benchmark 上，`native`、`native_mt`、`native_mt_stream` 仍会 xfail，因此这里暂时不能把它们当作稳定的 nonlinear sweep 基线。

不过现在已经具备两项关键能力：

1. `scripts/run_cudss_iteration_scan.py` 会直接把 `mt_mode/stream_mode` 一起落表；
2. native C++ 路径新增了 `DEVSIM_CUDSS_USE_STREAM=1` 和 `stream_mode=1/0` 统计，后续一旦 native nonlinear case 稳定，就可以直接确认 MT/stream 是否真的命中。

下一候选 benchmark：

- `examples/mobility/gmsh_mos2d.py`
- `testing/mos_2d.py`

原因是它们同时具备：

1. 非线性 drift-diffusion；
2. 多次 DC / bias 过程；
3. 比 diode benchmark 更大的矩阵规模，更有机会真正体现 cuDSS speedup。

## 10. MT mode / stream mode 现阶段结论

1. 用户前面关注的百万级 `2.582x solve-only` 结果走的是 `DEVSIM_CUDSS_DIRECT_SOLVER=custom`，**不是 native MT mode**。
2. `DEVSIM_CUDSS_MT_MODE=1` 只在 native C++ 路径尝试启用；只有 transfer stats 里打印出 `mt_mode=1`，才算真的命中。
3. 本轮新增 `DEVSIM_CUDSS_USE_STREAM=1`：
   - 只对 native C++ 路径生效；
   - 命中时 transfer stats 会打印 `stream_mode=1`。
4. 因而现在可以明确区分四种状态：
   - custom callback：无 native `mt_mode/stream_mode`
   - native：`mt_mode=0`, `stream_mode=0`
   - native + MT：`mt_mode=1`
   - native + MT + stream：`mt_mode=1`, `stream_mode=1`
5. 参考 NVIDIA `simple_multithreaded_mode.cpp` 后，本地实现已与 sample 对齐的关键点包括：
   - `cudssCreate(handle)` 后显式配置 MT threading layer；
   - 可选 `cudssSetStream(handle, stream)`；
   - profile 中把 `mt_mode/stream_mode` 明确打印出来，避免“以为开了其实没命中”。
6. 当前环境下用 `testing/cap2` 验证得到：
   - native：`stream_mode=0 mt_mode=0`
   - native + MT：`stream_mode=0 mt_mode=1`
   - native + MT + stream：`stream_mode=1 mt_mode=1`
7. MT mode 在当前 `libcudss_mtlayer_gomp.so.0` runtime 下，显式 teardown 会触发退出阶段的 `libgomp` 崩溃；当前实现已改为 **MT 模式下把 cuDSS 对象保留到进程退出统一回收**，从而保证用例正常结束。

### 10.1 当前 MT 基线（`cap2d_large`, `CAP2D_LARGE_MESH_SCALE=0.1`）

先明确口径：

- 历史 `2.582x` 是 **custom 路径的 solve-only speedup**
- 本节对比的是 native 路径自己的 `total` / `solve-only` 基线

先前未修复 `Device::Update` 时，native 路径的基线是：

| mode | total speedup | solve-only speedup | 结论 |
| --- | ---: | ---: | --- |
| native | `1.380x` | `1.761x` | 旧 native 基线 |
| native + MT | `1.517x` | `2.054x` | MT 确实提高了 native 加速比 |
| native + MT + stream | `1.143x` | `2.035x` | stream 当前未带来 total 正收益 |
| 历史 custom | - | `2.582x` | 当时仍高于 native + MT solve-only |

本轮修复 native `Device::Update` 默认路径后，`native + MT` 的当前结果更新为：

- 默认配置：`DEVSIM_CUDSS_BACKEND_POLICY=native`
- `DEVSIM_CUDSS_MT_MODE=1`
- `DEVSIM_CUDSS_USE_STREAM=0`
- `DEVSIM_CUDSS_DEVICE_UPDATE_STRATEGY` 不设置（默认）

3 次复测中位数：

| mode | total speedup | solve-only speedup | 说明 |
| --- | ---: | ---: | --- |
| native + MT（当前默认） | `2.017x` | `4.408x` | 已高于历史 custom `2.582x` solve-only |

### 10.2 `Device::Update` 热点定位与 bulk-copy 修复结果

根因定位：

1. native 路径里，`EquationHolder -> Equation::TryUpdateFromDevice -> copy_rows_to_host_double` 会对每个 row 单独做一次 `cudaMemcpy(sizeof(double))`；
2. 因此此前 `native + MT` 虽然已经把 `linear_solve` 压下来了，但 `Device::Update` 仍会膨胀到 `14.59s` 量级；
3. MT 本身没有把别的阶段变慢，真正的问题是 native 结果回传后的逐行 D2H。

`cap2d_large(scale=0.1)` 上的 A/B 结果：

| device update strategy | total speedup | solve-only speedup | 结论 |
| --- | ---: | ---: | --- |
| `rows` | `1.210x` | `1.596x` | 逐行 `cudaMemcpy`，最慢 |
| `fullcopy`（显式） | `1.901x` | `3.813x` | bulk D2H 后再 host 切片，显著更好 |
| 默认（本轮修复后） | `2.017x` | `4.408x` | 默认已命中 bulk-copy 路径 |

当前 native + MT profile（一次代表性复测）：

- `iteration_linear_solve_seconds = 2.61097s`
- `iteration_load_dc_seconds = 9.12845s`
- `iteration_device_update_seconds = 0.05006s`

也就是：

1. `Device::Update` 已从先前的 `14.59s` 降到约 `0.05s`；
2. 证明之前的差距主要不是 MT 未生效，而是逐行 D2H；
3. bulk-copy 修复后，native + MT 的 solve-only 已经反超历史 custom 基线。

### 10.3 同一 case 两次迭代逐项对比（baseline / 历史 custom / 当前 native+MT）

#### baseline（不用 cuDSS）

| iter | total(s) | load_dc(s) | linear_solve(s) | device_update(s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | `25.1525` | `7.9419` | `16.1869` | `0.0127` |
| 1 | `13.6544` | `1.0191` | `12.5671` | `0.0100` |

#### 历史 custom（`2.582x solve-only` 基线）

| iter | total(s) | load_dc(s) | linear_solve(s) | device_update(s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | `12.5541` | `8.1054` | `3.2017` | `0.2840` |
| 1 | `1.6912` | `1.0792` | `0.2554` | `0.2876` |

#### 当前 native + MT（本轮修复后，一次代表性复测）

| iter | total(s) | load_dc(s) | linear_solve(s) | device_update(s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | `11.4668` | `7.9977` | `2.4274` | `0.0228` |
| 1 | `1.3997` | `1.1307` | `0.1836` | `0.0272` |

结论：

1. baseline（不用 cuDSS）的两次迭代主要耗时始终在 `linear_solve`；
2. 历史 custom 和当前 native + MT 在 `LoadMatrixAndRHS` 上已经是同量级；
3. native + MT 修复后，`Device::Update` 不再是热点，且两次迭代的 `linear_solve` 都已低于历史 custom。

### 10.4 当前 native + MT 推荐组合（已有点测）

`cap2d_large(scale=0.1)` 上的点测结果：

| mode / config | total speedup | solve-only speedup | 备注 |
| --- | ---: | ---: | --- |
| native + MT（当前默认） | `2.017x` | `4.408x` | 当前推荐基线 |
| native + MT + `HOST_NTHREADS=8` | `1.376x` | `1.759x` | 比 default 更差 |
| native + MT + `HYBRID_MODE=1` | `1.484x` | `1.994x` | 略低于 default |
| native + MT + `REORDERING_ALG=1` | fail | fail | `FACTORIZATION status=2` |
| native + MT + stream | `1.143x` | `2.035x` | 当前不建议默认开启 |

结论：在当前已测点里，**`native + MT + 默认 bulk-copy 更新路径` 就是当前推荐组合**；下一轮调参应继续围绕“受支持的 config 组合”做系统扫描，而不是默认打开 stream / host_threads / hybrid。

## 11. 对 `cvxgrp/scs`、`CUDA_Newton_project` 与后续优化的结论

1. `cvxgrp/scs` 借鉴点是 cuDSS phase 管理：
   - `ANALYSIS + FACTORIZATION`
   - `REFACTORIZATION`
   - pinned staging
   - diagnostics/inertia
   它不是 DEVSIM 外层 Newton 方法的直接模板。
2. `CUDA_Newton_project` 更像教学/benchmark 工程：
   - Jacobian 构造方式与 DEVSIM 不同；
   - 每轮仍偏向完整 factorization；
   - 缺少 DEVSIM 当前需要的稳健 Newton 外层机制；
   因此不适合作为主模板。
3. 现阶段真正值得继续推进的优先级：
   - 第一优先级：`LoadMatrixAndRHS / Finalize / Device::Update` 这类 solver framework 开销；
   - 第二优先级：native `MT + configset/hybrid/reordering` 组合收敛；
   - 第三优先级：继续扩大 nonlinear benchmark，观察迭代次数增长后的 speedup 曲线。
