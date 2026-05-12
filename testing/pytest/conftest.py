from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from case_parser import CaseSpec, filter_cases, load_cases


def _extra_benchmark_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="examples/capacitance/cap2d_large",
            script="cap2d_large.py",
            script_args=(),
            working_dir="/devsim/examples/capacitance",
            depends=(),
        ),
        CaseSpec(
            name="examples/diode/diode_1d_cudss_bench",
            script="diode_1d_cudss_bench.py",
            script_args=(),
            working_dir="/devsim/examples/diode",
            depends=(),
        ),
        CaseSpec(
            name="examples/mobility/gmsh_mos2d",
            script="gmsh_mos2d.py",
            script_args=(),
            working_dir="/devsim/examples/mobility",
            depends=(),
        ),
        CaseSpec(
            name="testing/mos_2d",
            script="mos_2d.py",
            script_args=(),
            working_dir="/devsim/testing",
            depends=(),
        )
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("devsim-cudss")
    group.addoption(
        "--cmake-file",
        action="store",
        default="",
        help="Path to testing/CMakeLists.txt (defaults to /devsim/testing/CMakeLists.txt)",
    )
    group.addoption(
        "--ctest-file",
        action="store",
        default="",
        help="Deprecated alias for --cmake-file",
    )
    group.addoption(
        "--solver-mode",
        action="store",
        default="both",
        choices=("baseline", "cudss", "both"),
        help="Run baseline only, cudss only, or both (default).",
    )
    group.addoption(
        "--case-filter",
        action="store",
        default="",
        help="Regex to filter case names.",
    )
    group.addoption(
        "--max-cases",
        action="store",
        type=int,
        default=0,
        help="Limit number of parametrized cases for quick debug.",
    )
    group.addoption(
        "--devsim-so",
        action="store",
        default="",
        help="Path to devsim_py3 shared module for deterministic runner.",
    )
    group.addoption(
        "--strict-cudss",
        action="store_true",
        default=False,
        help="Fail test on cuDSS run failures instead of xfail.",
    )
    group.addoption(
        "--print-timing",
        action="store_true",
        default=False,
        help="Print baseline vs cuDSS timing summary per case when both modes run.",
    )
    group.addoption(
        "--timing-json",
        action="store",
        default="",
        help="Optional JSONL output path for per-case timing details.",
    )


def _default_cmake_file() -> Path:
    cmake_file = Path("/devsim/testing/CMakeLists.txt")
    if cmake_file.is_file():
        return cmake_file
    raise RuntimeError("No testing/CMakeLists.txt found")


def _has_cudss_gpu(devsim_so: str) -> bool:
    code = (
        "import sys; "
        "sys.path.insert(0, '/devsim'); "
        "from cudss.cudss_shim import local_solver_callback; "
        "ret=local_solver_callback(action='init', n=1, transpose=False); "
        "print('1' if ret.get('status') else '0')"
    )
    env = None
    cmd = [sys.executable, "-c", code]
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return cp.returncode == 0 and cp.stdout.strip().endswith("1")


def pytest_configure(config: pytest.Config) -> None:
    cmake_opt = config.getoption("--cmake-file") or config.getoption("--ctest-file")
    cmake_file = Path(cmake_opt) if cmake_opt else _default_cmake_file()
    all_cases = load_cases(cmake_file)
    all_cases.extend(_extra_benchmark_cases())
    config._devsim_case_index_all = {c.name: c for c in all_cases}  # type: ignore[attr-defined]
    cases = filter_cases(all_cases, config.getoption("--case-filter") or None)
    max_cases = int(config.getoption("--max-cases") or 0)
    if max_cases > 0:
        cases = cases[:max_cases]
    config._devsim_cases = cases  # type: ignore[attr-defined]

    devsim_so = config.getoption("--devsim-so")
    config._devsim_so = devsim_so  # type: ignore[attr-defined]
    config._cudss_available = _has_cudss_gpu(devsim_so)  # type: ignore[attr-defined]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_spec" in metafunc.fixturenames:
        cases: list[CaseSpec] = metafunc.config._devsim_cases  # type: ignore[attr-defined]
        metafunc.parametrize("case_spec", cases, ids=[c.name for c in cases])


@pytest.fixture(scope="session")
def solver_mode(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--solver-mode"))


@pytest.fixture(scope="session")
def devsim_so(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig._devsim_so)  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def cudss_available(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig._cudss_available)  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def strict_cudss(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--strict-cudss"))


@pytest.fixture(scope="session")
def case_index(pytestconfig: pytest.Config) -> dict[str, CaseSpec]:
    return dict(pytestconfig._devsim_case_index_all)  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def print_timing(pytestconfig: pytest.Config) -> bool:
    return bool(pytestconfig.getoption("--print-timing"))


@pytest.fixture(scope="session")
def timing_json(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--timing-json"))
