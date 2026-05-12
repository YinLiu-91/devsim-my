# DEVSIM cuDSS 接入架构设计

配套使用说明见：`doc/cudss_usage.md`
配套优化方法论见：`doc/cudss_optimization_skill.md`

## 1. 文档范围

本文档只描述**当前主线已落地**的 cuDSS 接入架构，范围以最近两次接入相关提交为准：

1. `6af1978 Integrate cuDSS native MT workflow`
2. `ff1bb4e Finalize cuDSS callback regression cleanup`

本文档关注的是**架构设计**，不是命令手册；因此重点回答：

1. DEVSIM 是如何把 cuDSS 接进现有 direct solver 框架的；
2. native C++ 与 callback/shim 两条后端路径如何在统一接口下协同；
3. 求解结果如何在 host / device 两侧流转并被 Newton/Device/Equation 消费；
4. 为什么 callback 路径里还保留 residual / verify / UMFPACK fallback；
5. 当前方案的特点、约束、边界与后续扩展点是什么。

## 2. 背景与设计目标

在这轮接入前，DEVSIM 已经支持通过 `solver_callback` 接入外部 direct solver，但 cuDSS 仍主要停留在 Python callback 热路径里。当前方案的设计目标是：

1. **统一入口**：继续使用 `direct_solver` / `solver_callback` 这一套既有接口，不重新发明一套求解器 API。
2. **双后端并存**：在 `direct_solver="cudss"` 下，同时支持 native C++ cuDSS 后端和 callback/shim 后端。
3. **性能路径前置**：优先收敛到 native C++ + MT 这条默认主线，绕开 Python/ctypes 热路径。
4. **正确性优先**：即使 native 不可用，或 callback 路径出现数值异常，也必须有明确 fallback 和回归验证手段。
5. **结果延迟物化**：允许求解结果先保留在 device 侧，只有在 Newton/Device/Equation 真的需要 host 向量时才回传。

## 3. 非目标

当前设计**没有**试图一次性解决以下问题：

1. 把所有求解流程都搬到 GPU；
2. 替换 DEVSIM 现有的所有 custom solver 机制；
3. 在 native cuDSS 路径里完整支持 AC / Noise / Transient / complex solve；
4. 改造矩阵组装主链（`LoadMatrixAndRHS`）为 device-native。

因此它是一个**分阶段接入方案**：先把 cuDSS 稳定接上，再让 native+MT 成为默认性能主线，同时保留 callback 作为兼容和 correctness 收口路径。

## 4. 总体架构

### 4.1 逻辑分层

```text
用户脚本 / pytest / shell scripts
        |
        | set_parameter(direct_solver, solver_callback, cudss_result_mode)
        v
SolverUtil / Preconditioner 工厂
        |
        +--> direct_solver="custom" --> ExternalPreconditioner
        |
        \--> direct_solver="cudss" --> CuDSSPreconditioner
                                         |
                                         +--> native C++ cuDSS backend
                                         |
                                         \--> Python callback/shim backend
                                                  |
                                                  \--> cudss_loader + cudss_shim
        |
        v
ResultView / DeviceResultBuffer
        |
        v
Newton -> Device::Update -> EquationHolder -> Equation::TryUpdateFromDevice
```

### 4.2 核心设计结论

这次接入不是简单“加了一个新的 Python callback”，而是做了三件事：

1. 在 `src/math/` 里引入了**统一的 cuDSS 门面**：`CuDSSPreconditioner`；
2. 在 `ResultView` / `DeviceResultBuffer` 上建立了**host/device 双态结果契约**；
3. 在 `EquationHolder` / `Equation` 里打通了**device result 的消费链**。

## 5. 核心模块职责

| 模块 | 关键文件 | 责任 |
| --- | --- | --- |
| 求解器选择层 | `src/math/SolverUtil.cc` | 根据 `direct_solver` 选择 `ExternalPreconditioner` 或 `CuDSSPreconditioner` |
| 统一抽象层 | `src/math/Preconditioner.hh/.cc` | 定义 `LUFactor/LUSolve`、`GetResultView()`、device result 基础接口 |
| cuDSS 门面层 | `src/math/CuDSSPreconditioner.hh/.cc` | 承载 `direct_solver="cudss"` 主入口，内部再分发到 native 或 callback |
| 通用 callback 层 | `src/math/ExternalPreconditioner.hh/.cc` | 保留历史 `solver_callback` 通道，也承接 callback 结果视图/profile |
| Python runtime 发现层 | `cudss/cudss_loader.py` | 探测 Python binding 或共享库 runtime |
| callback 可靠性控制层 | `cudss/cudss_shim.py` | 实现 init/factor/solve/gather_rows/stats，带 residual/verify/UMFPACK fallback |
| 结果消费层 | `src/math/dsMathTypes.hh` | 定义 `ResultView` 和 `DeviceResultBuffer` |
| Newton 消费层 | `src/math/Newton.cc` | solve 后按 `ResultView` 决定 host/device 更新路径 |
| 方程更新层 | `src/Equation/EquationHolder.cc` / `src/Equation/Equation.cc` | full copy 或 row gather 更新 node model |
| 回归框架 | `testing/pytest/run_case.py` / `test_cudss_compare.py` | baseline vs cuDSS 的统一执行与对比 |
| 工程化入口 | `scripts/cudss_common.sh` / `run_cudss_pytest_regression.sh` | 固化 native+MT 默认运行口径 |

## 6. 统一入口设计

### 6.1 Python/脚本侧统一入口

当前推荐入口仍是：

```python
devsim.set_parameter(name="direct_solver", value="cudss")
devsim.set_parameter(name="solver_callback", value=local_solver_callback)
```

这里有两个关键点：

1. `direct_solver="cudss"` 决定走 `CuDSSPreconditioner`；
2. `solver_callback` 仍然保留，即使最终可能走 native C++，因为：
   - callback 是兼容 fallback；
   - callback 仍是当前 correctness 收口与对照路径；
   - `auto` 策略下 native 初始化失败时要能退回 callback。

### 6.2 C++ 侧选择逻辑

`src/math/SolverUtil.cc` 中：

1. `direct_solver="custom"` -> `ExternalPreconditioner`
2. `direct_solver="cudss"` -> `CuDSSPreconditioner`

也就是说，**`cudss` 不是把原有 custom 通道替换掉，而是在原工厂体系里新增了一个更高层的 cuDSS 门面**。

## 7. `CuDSSPreconditioner`：统一 cuDSS 门面

### 7.1 为什么需要这一层

`CuDSSPreconditioner` 的作用不是“再包一层 callback”，而是把以下能力收敛到一个类里：

1. native / callback 后端策略选择；
2. cuDSS profile 统计；
3. symbolic reuse / refactor 生命周期管理；
4. device result 暴露；
5. host materialize 与 device-experimental 的行为统一。

### 7.2 后端策略

它通过 `DEVSIM_CUDSS_BACKEND_POLICY` 选择后端：

- `native`：强制 native C++ 路径；
- `callback`：强制 callback/shim 路径；
- `auto`：优先 native，不可用时 fallback 到 callback。

同时继续兼容旧开关：

- `DEVSIM_CUDSS_NATIVE_CPP=1/0`

### 7.3 native 初始化流程

`CuDSSPreconditioner::init()` 在 native 路径中做的事包括：

1. 动态加载 `libcudss` / `libcudart`；
2. 解析 cuDSS / CUDA 关键符号；
3. 探测 GPU；
4. 可选打开 stream；
5. 可选设置 MT threading layer；
6. 创建 `config` / `data`；
7. 注册 `DeviceResultBuffer` 的 D2H / row-gather 回调。

如果这些步骤失败：

- 在 `native` 强制模式下直接报错；
- 在 `auto` 模式下切回 callback。

### 7.4 factor / solve 生命周期

native 路径下：

1. **首次 factor 或 symbolic 改变**
   - 分配并上传 `Ap/Ai/Ax`；
   - 创建 cuDSS CSR / dense matrix 句柄；
   - 执行 `ANALYSIS + FACTORIZATION`。
2. **symbolic 未变**
   - 只更新 `Ax`；
   - 执行 `REFACTORIZATION`。
3. **solve**
   - 上传 RHS；
   - 执行 `SOLVE`；
   - 根据 `cudss_result_mode` 和 `NodeKeeper` 决定：
     - 直接 D2H 物化 host 向量；
     - 或保留 device result token/buffer，延迟到后续消费。

### 7.5 symbolic 复用设计

当前方案支持两级 symbolic reuse 判断：

1. 常规路径：依赖 `CompressedMatrix` 报告的 `SymbolicStatus_t::SAME_SYMBOLIC`；
2. 可选增强：通过 `DEVSIM_CUDSS_SYMBOLIC_HASH_REUSE` 基于 `Ap/Ai` 哈希做额外 pattern 复用判断。

这使它既能遵守现有矩阵框架语义，又为后续更激进的 pattern reuse 留出空间。

## 8. callback/shim 架构

### 8.1 为什么 callback 路径仍然保留

callback 路径现在不是默认性能主线，但仍是架构里的必要组成：

1. native 不可用时的后备方案；
2. pytest compare 的历史对照路径；
3. 复杂 fallback / correctness 收口逻辑目前主要沉淀在 shim 层；
4. 对 native 行为做独立对照时，callback 是非常有价值的参照系。

### 8.2 shim 的职责不是“薄封装”

`cudss/cudss_shim.py` 当前承担了完整的 callback 后端状态机：

1. `init`
2. `factor`
3. `solve`
4. `gather_rows`
5. `stats`

同时它还负责：

1. runtime 检测与 context 复用；
2. cuDSS config set 和 unsupported 场景回退；
3. pinned staging / zero-copy experiment；
4. host-side CSR cache；
5. residual 守卫；
6. verify fallback；
7. UMFPACK second opinion / fallback。

因此，**shim 在当前架构中的角色是“callback 可靠性控制层”，不是简单 API 绑定层**。

### 8.3 callback 的 factor / solve 设计

factor 阶段：

1. 首次或 symbolic 改变时上传 `Ap/Ai/Ax`；
2. 结构不变时只更新 `Ax`；
3. 执行 `ANALYSIS + FACTORIZATION` 或 `REFACTORIZATION`；
4. 在 `NOT_SUPPORTED + tuned config` 下可自动回退到默认 config 重试。

solve 阶段：

1. 上传 RHS；
2. 执行 `SOLVE`；
3. 根据 `result_mode` 和 `require_host_x` 决定：
   - 返回 host `x`；
   - 或仅返回 `x_device_token` / `x_location`，并通过 `gather_rows` 暴露局部回读能力。

### 8.4 correctness fallback 设计

callback 路径的关键价值在于它显式实现了两层数值保护：

1. **residual fallback**
   - 对 `Ax-b` 做无穷范数检查；
   - 超阈值时用 UMFPACK 重算并替换结果。
2. **verify fallback**
   - 在前几次 solve、或残差相对 RHS 偏大时，引入 UMFPACK second opinion；
   - 若 fallback 残差显著更好，则切换到 fallback 结果。

这部分逻辑正是最后一个 commit 要保留的核心价值：它直接决定 callback compare 能否稳定收口。

## 9. 结果视图与 device result 消费链

### 9.1 统一结果契约

`src/math/dsMathTypes.hh` 中新增：

- `DeviceResultBuffer`
- `ResultView<T>`

它们把求解结果从“只有 host `x` 向量”扩展为“**host payload + 可选 device result**”的统一契约。

`ResultView` 中最关键的字段有：

| 字段 | 含义 |
| --- | --- |
| `host_values` | 已经可直接消费的 host 向量 |
| `host_payload` | 保存在 `ObjectHolder` 中的 host 结果 |
| `device_result` | 指向可回读的 device result buffer |
| `device_token` | 本次 solve 的结果标识 |
| `location` | `host` / `device_experimental_*` 等位置标签 |

### 9.2 Newton 侧消费方式

`src/math/Newton.cc` 在 linear solve 完成后：

1. 从 preconditioner 取 `ResultView`；
2. 若结果位置是 `device_experimental*`，记录本轮确实观察到了 device result；
3. 若当前流程必须要 host 向量，则回退到 host 物化；
4. 把 `ResultView` 直接传给 `Device::Update(result_view)`。

因此 Newton 不再假设“solve 一定立即给我一个完整 host 向量”，而是允许后端暴露**延迟物化结果**。

### 9.3 EquationHolder / Equation 的消费策略

`EquationHolder.cc` 当前支持两种 device result 消费方式：

1. **FULLCOPY**
   - 先整向量 D2H；
   - 再按传统 host 路径更新。
2. **ROWS**
   - 只回读当前 equation 需要的行；
   - 通过 `Equation::TryUpdateFromDevice()` 把 equation row 映射到 node index，再应用 update rule。

由环境变量控制：

```bash
DEVSIM_CUDSS_DEVICE_UPDATE_STRATEGY=fullcopy|rows|auto
```

当前默认是 `fullcopy`，原因很明确：它更稳、更简单，且此前已经证明逐行 D2H 会造成严重额外开销；`rows` 主要保留为显式诊断/实验路径。

### 9.4 设计收益

这条消费链的意义在于：

1. native 和 callback 都可以通过同一 `ResultView` 暴露结果；
2. Newton/Device/Equation 无需知道具体求解后端；
3. host/device 结果消费策略被显式化，而不是散落在求解器内部。

## 10. 配置面设计

当前配置面分为三层。

### 10.1 主配置面

这些是当前架构文档推荐重点关注的参数：

| 参数 | 作用 |
| --- | --- |
| `DEVSIM_CUDSS_BACKEND_POLICY` | 选择 `auto/native/callback` |
| `DEVSIM_CUDSS_MT_MODE` | 控制 native MT mode |
| `DEVSIM_CUDSS_RESULT_MODE` | 控制 host / device-experimental 结果模式 |
| `DEVSIM_CUDSS_DIRECT_SOLVER` | pytest/case runner 中决定 `custom` 还是 `cudss` |
| `DEVSIM_CUDSS_PROFILE` | 打开 cuDSS 侧 profile/transfer stats |

### 10.2 兼容与运行时发现

| 参数 | 作用 |
| --- | --- |
| `DEVSIM_CUDSS_LIB` | 指定 `libcudss` 路径 |
| `DEVSIM_CUDART_LIB` | 指定 `libcudart` 路径 |
| `DEVSIM_CUDSS_NATIVE_CPP` | 旧开关，兼容 native/callback 选择 |

### 10.3 高级/实验配置面

这些参数存在，但不应当作为默认主线配置：

- `DEVSIM_CUDSS_USE_STREAM`
- `DEVSIM_CUDSS_REORDERING_ALG`
- `DEVSIM_CUDSS_HYBRID_MODE`
- `DEVSIM_CUDSS_HOST_NTHREADS`
- `DEVSIM_CUDSS_HYBRID_EXECUTE_MODE`
- `DEVSIM_CUDSS_SYMBOLIC_HASH_REUSE`
- `DEVSIM_CUDSS_PINNED_STAGING`
- `DEVSIM_CUDSS_DEVICE_UPDATE_STRATEGY`
- shim 内的 residual / verify / zero-copy 相关实验开关

当前架构思路是：**默认路径尽量少变量，实验参数留给专项 profiling 和诊断。**

## 11. 测试与回归设计

### 11.1 为什么测试也属于架构的一部分

这次接入不是单点求解器替换，而是引入了：

1. 新的后端策略；
2. 新的结果位置语义；
3. callback/native 两套执行实现；
4. correctness fallback。

因此测试框架本身就是架构的一部分。

### 11.2 pytest compare 架构

`testing/pytest/run_case.py` 负责：

1. 统一加载 `devsim`；
2. 统一按 `solver-mode` 注入 baseline / cuDSS；
3. 记录 timing json；
4. 在 cuDSS custom 路径下采集 shim 统计。

`testing/pytest/test_cudss_compare.py` 负责：

1. 先跑 baseline；
2. 再跑 cuDSS；
3. 对输出做带浮点容差的比较；
4. 过滤 solver-only 诊断噪音；
5. 对当前 phase-1 不支持的 case 做 skip，对未收口 case 做 xfail。

这保证了当前架构的验证口径不是“只要能跑就算接入成功”，而是：

1. correctness 可对照；
2. 性能可测量；
3. 限制和失败模式可显式表达。

### 11.3 shell 入口的角色

`scripts/cudss_common.sh` 把 native+MT 默认环境固化为一套稳定入口：

```bash
DEVSIM_CUDSS_BACKEND_POLICY=native
DEVSIM_CUDSS_MT_MODE=1
DEVSIM_CUDSS_USE_STREAM=0
DEVSIM_CUDSS_RESULT_MODE=device_experimental
DEVSIM_CUDSS_DIRECT_SOLVER=cudss
```

这使人工执行时不需要记忆一长串环境变量，也让“当前默认主线”在脚本层面是可重放、可复现的。

## 12. 当前方案的特点

### 12.1 优点

1. **入口统一**：用户侧仍是熟悉的 `direct_solver` / `solver_callback` 心智模型。
2. **分层清楚**：solver 选择、后端实现、结果消费、回归验证各层职责明确。
3. **双后端兼容**：native 负责默认性能主线，callback 负责兼容和 correctness 收口。
4. **结果契约显式化**：通过 `ResultView` / `DeviceResultBuffer`，host/device 结果不再混杂。
5. **可观测性增强**：native 与 callback 都暴露 profile / transfer stats。

### 12.2 代价与 trade-off

1. 架构更复杂：现在不是单一路径，而是 `custom`、`cudss/native`、`cudss/callback` 并存。
2. callback shim 逻辑较重：它承担了大量可靠性控制，而非简单 API 适配。
3. 求解结果仍常常需要 host 物化：因为 Newton/Node/Equation 仍大量依赖 host 侧更新。
4. MT 模式的 native teardown 需要特殊处理：当前通过延后清理规避 shutdown crash。

## 13. 已知约束与边界

1. native cuDSS 当前只支持**实数 solve 主路径**；complex/native 仍未打通。
2. 当前接入重点仍是 DC real-valued 路径，AC/Noise/Transient 不是这轮接入主目标。
3. callback compare 已基本收口，但 native+MT pytest 仍有历史 `xfail/skip`，因此 native 主线还不是“所有 case 全支持”。
4. 当前总体性能瓶颈已不主要在 `linear_solve`，而更多转向 `LoadMatrixAndRHS`/assembly 前端；这属于求解器接入后的下一层问题，不是 cuDSS 接入层自身缺失。

## 14. 后续扩展方向

从当前架构出发，最自然的后续方向有：

1. **扩展 native 覆盖面**
   - complex/AC 支持；
   - 更完整的 phase 覆盖。
2. **继续压缩 host 依赖**
   - 减少必须回主机的结果物化；
   - 改善 `ResultView` 到 equation update 的消费效率。
3. **统一 native/callback 观测面**
   - 让 profile、fallback、统计字段更一致。
4. **推进更深层性能优化**
   - 不是继续只抠 cuDSS config，而是继续往 assembly / device update 主链深入。

## 15. 一句话总结

当前 DEVSIM 的 cuDSS 接入架构，本质上是一个**“统一入口 + 双后端 + 显式结果视图 + correctness fallback + 回归工程化”**的分层方案：它既把 native+MT 收敛成默认性能主线，又保留了 callback 路径在兼容性与正确性上的工程价值。
