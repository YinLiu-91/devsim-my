# HOWTO
## Update version number

To update the version number update the lines in these files:

* ``CMakeLists.txt``
```
ADD_DEFINITIONS(-DDEVSIM_VERSION_STRING=\"2.10.0\")
```
* ``dist/bdist_wheel/setup.cfg``
```
version = 2.9.2
```

## Update minimum python version
* ``dist/bdist_wheel/setup.cfg``
```
py-limited-api = cp39
```
* ``src/pythonapi/CMakeLists.txt``
```
target_compile_definitions(pythonapi_interpreter_py3 PRIVATE -DDEVSIM_MODULE_NAME=devsim_py3 -DPy_LIMITED_API=0x03090000)
```
 * Additional Python Notes
   * [how-to-configure-setuptools-with-setup-cfg-to-include-platform-name-python-tag](https://stackoverflow.com/questions/72090919/how-to-configure-setuptools-with-setup-cfg-to-include-platform-name-python-tag)
   * [C API Stability](https://docs.python.org/3/c-api/stable.html)
   * It looks like ``setup.cfg`` is going away, but it is not clear what to do for the replacement ``pyproject.toml``.

## Debug `bioapp1_2d.py` solve stage with gdb

This reproduces the solve-path debugging flow for:
`/devsim/examples/bioapp1/bioapp1_2d.py`

### 1. Run under gdb and break at `solve`

From repo root:

```bash
gdb -q --batch \
  -ex 'set pagination off' \
  -ex 'set breakpoint pending on' \
  -ex 'set env PYTHONPATH /devsim/dist/devsim_ubuntu_18.04/lib' \
  -ex 'cd /devsim/examples/bioapp1' \
  -ex 'break dsCommand::solveCmd(CommandHandler&)' \
  -ex 'run bioapp1_2d.py 7' \
  -ex 'bt 12' \
  -ex 'continue' \
  -ex 'bt 12' \
  --args python3
```

Notes:
- `set breakpoint pending on` is required because `devsim_py3.so` is loaded dynamically after Python starts.
- This case calls `solve(...)` twice, so the breakpoint is hit twice.

### 2. What the backtrace proves

When stopped at the breakpoint, the top of the stack shows:

1. `dsCommand::solveCmd(CommandHandler&)` in `devsim_py3.so`
2. `dsPy::CmdDispatch(...)` in `devsim_py3.so`
3. Python runtime frames (`_PyEval_EvalFrameDefault`, etc.)

This confirms Python `solve(...)` enters the C++ command implementation.

### 3. Source files for the solve call chain

- Python import: `dist/devsim_ubuntu_18.04/lib/devsim/__init__.py`
  - `from .devsim_py3 import *`
- Command registration: `src/pythonapi/CommandTable.cc`
  - `DS_FUNCTION_TABLE(solve, dsCommand::solveCmd)`
- Python/C dispatch: `src/pythonapi/PythonCommands.cc` (`CmdDispatch`)
- Solve command implementation: `src/commands/MathCommands.cc` (`solveCmd`)
- Newton solve loop: `src/math/Newton.cc` (`Newton::Solve`)

### 4. Optional interactive debugging

If you want to inspect arguments/variables interactively:

```bash
gdb --args python3 /devsim/examples/bioapp1/bioapp1_2d.py 7
```

Then in gdb:

```gdb
set env PYTHONPATH /devsim/dist/devsim_ubuntu_18.04/lib
set breakpoint pending on
break dsCommand::solveCmd(CommandHandler&)
run
bt
continue
```

## Run pytest baseline-vs-cuDSS comparison

Pytest entrypoint for Python case migration and cuDSS comparison:

```bash
pytest -q testing/pytest --devsim-so=/devsim/linux_x86_64_debug/src/main/devsim_py3.so
```

Common options:

```bash
# baseline only
pytest -q testing/pytest --solver-mode=baseline

# cudss only
pytest -q testing/pytest --solver-mode=cudss

# both (default): run baseline then cudss and compare outputs
pytest -q testing/pytest --solver-mode=both

# filter cases
pytest -q testing/pytest --case-filter='testing/cap2|testing/mesh2d'

# quick debug with first N cases
pytest -q testing/pytest --max-cases=10

# strict mode: fail immediately on cuDSS execution failures or output mismatches
pytest -q testing/pytest --strict-cudss

# print per-case timing summary (baseline total, cuDSS total, speedup, delta)
pytest -q testing/pytest --solver-mode=both --print-timing --case-filter='testing/cap2'

# append machine-readable timing JSONL report
pytest -q testing/pytest --solver-mode=both --timing-json=/tmp/cudss_timing.jsonl --case-filter='testing/cap2'
```

Behavior notes:

- `--solver-mode`:
  - `baseline`: baseline only
  - `cudss`: cudss only (skip when cudss runtime/GPU unavailable)
  - `both` (default): baseline then cudss; compare normalized outputs; emit timing summary/json when requested

- `skip`:
  - cuDSS runtime/GPU unavailable (`--solver-mode=cudss` or `both`)
  - case is outside Phase-1 scope (AC/Noise/Transient)
- `xfail` (default, non-strict):
  - cuDSS execution failed
  - baseline/cudss output mismatch after float-tolerant normalization
- `--strict-cudss`:
  - above `xfail` conditions are promoted to hard fail
- `run_case.py` solver switch details:
  - cudss path always sets:
    - `direct_solver=custom`
    - `solver_callback=cudss.cudss_shim.local_solver_callback`
  - baseline path only sets UMFPACK callback when current `direct_solver == "unknown"`:
    - `direct_solver=custom`
    - `solver_callback=umfpack.umfshim.local_solver_callback`
- timing metrics (`--print-timing` / `--timing-json`):
  - process wall-time is recorded as total time (`baseline_total_s`, `cudss_total_s`)
  - `speedup = baseline_total_s / cudss_total_s`, `delta_s = baseline_total_s - cudss_total_s`
  - a Python-level `devsim.solve` wrapper reports approximate solver-attributable time
    (`baseline_solver_s`, `cudss_solver_s`, `solver_delta_s`) and solve call counts
  - non-solver estimate is `total - solver` (`baseline_nonsolver_s`, `cudss_nonsolver_s`)

Recommended validation workflow:

```bash
# 1) Baseline sanity
pytest -q testing/pytest --solver-mode=baseline --max-cases=10

# 2) Compare baseline vs cuDSS on filtered set
pytest -q testing/pytest --solver-mode=both --case-filter='testing/cap2|testing/mesh2d'

# 3) Strict gate on same filtered set
pytest -q testing/pytest --solver-mode=both --strict-cudss --case-filter='testing/cap2|testing/mesh2d'
```

Set this to trace cuDSS API calls:

```bash
export DEVSIM_CUDSS_DEBUG=1
```
