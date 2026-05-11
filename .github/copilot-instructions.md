# Copilot instructions for `devsim/devsim`

## Init

- 默认使用中文回答。
- Ubuntu 18.04 构建命令：
  ```bash
  bash scripts/build_ubuntu_18.04.sh devsim_ubuntu_18.04
  ```

## Build, test, and lint commands

### Build
```bash
# clone + submodules (required for external/symdiff, Eigen, UMFPACK, etc.)
git submodule init
git submodule update
```

```bash
# Linux wheel build used by release automation
bash scripts/build_manylinux_2_28.sh <version-tag>
```

```bash
# Ubuntu 18.04 build
bash scripts/build_ubuntu_18.04.sh devsim_ubuntu_18.04
```

```bash
# macOS wheel build
bash scripts/build_macos.sh <version-tag>
```

```bash
# Windows wheel build (from Anaconda prompt)
scripts\build_appveyor.bat x64 conda <version-tag>
```

```bash
# If build directories are already configured, compile directly
cd linux_x86_64_release && make -j4
```

### Tests
```bash
# full CTest suite (from a configured build dir)
cd linux_x86_64_release && ctest --output-on-failure
```

```bash
# run one test
cd linux_x86_64_release && ctest -R '^testing/cap2$' --output-on-failure
```

```bash
# direct smoke test from installation docs
cd testing && python cap2.py
```

### Lint / formatting
```bash
# same gate as CI
pre-commit run -a
```

```bash
# single-file lint/format
pre-commit run --files path/to/file.py
```

## High-level architecture

- Root `CMakeLists.txt` builds two top-level trees: `src/` (simulator + Python module) and `testing/` (CTest + regression harness).
- `src/` is split into static libraries by responsibility (`Geometry`, `models`, `Equation`, `math`, `meshing`, `Circuit`, etc.) and linked into the Python extension in `src/pythonapi/CMakeLists.txt` as `devsim_py3`.
- Python API flow is centralized in `src/pythonapi/PythonCommands.cc`: each Python call dispatches through `CmdDispatch` -> `CommandHandler`/`GetArgs` parsing -> `dsCommand::*` implementation in `src/commands/*.cc`.
- Command names are defined once in `src/pythonapi/CommandTable.cc` and reused to generate both dispatch functions and Python method table entries.
- Runtime state is global and shared via `GlobalData`; initialization in `src/main/devsim_py.cc` configures solver/math-library capabilities and default global entries.
- Regression tests are CTest entries from `testing/CMakeLists.txt`; they run Python scripts and compare generated outputs against `goldenresults/` using `testing/rundifftest.py`.

## Key codebase conventions

- DEVSIM Python commands are **keyword-argument driven**; positional arguments are rejected in `CmdDispatch`.
- Command option contracts are declared as static `dsGetArgs::Option[]` tables inside each command function, including required/optional flags, type, defaults, and validator callbacks.
- Validation logic is centralized in `src/commands/Validate.cc` and `CheckFunctions.cc`; commands typically accumulate an error string and return via `data.SetErrorResult(...)`.
- When adding/changing a command, update both:
  - `src/pythonapi/CommandTable.cc` (registration)
  - `src/pythonapi/DevsimDoc.cc` (user-visible command docstring)
- Test correctness is typically **exact text comparison** against golden outputs (`rundifftest.py`), so command output formatting and determinism matter.
- Version or Python API floor changes are coupled across files (see `HOWTO.md`): `CMakeLists.txt`, `dist/bdist_wheel/setup.cfg`, and `src/pythonapi/CMakeLists.txt`.
