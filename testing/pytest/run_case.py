from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any


def _load_devsim_module(devsim_so: str | None) -> None:
    if devsim_so and Path(devsim_so).is_file():
        spec = importlib.util.spec_from_file_location("devsim_py3", devsim_so)
        if not spec or not spec.loader:
            raise RuntimeError(f"Unable to load devsim module from {devsim_so}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["devsim"] = mod
        # Many example scripts import helpers as devsim.python_packages.*.
        # When loading the extension directly from a .so, expose that namespace.
        mod.__path__ = []  # type: ignore[attr-defined]
        import python_packages

        setattr(mod, "python_packages", python_packages)
        sys.modules["devsim.python_packages"] = python_packages
    else:
        import devsim  # noqa: F401


def _enable_cudss_solver() -> None:
    import devsim
    from cudss.cudss_shim import local_solver_callback

    direct_solver_value = os.environ.get("DEVSIM_CUDSS_DIRECT_SOLVER", "custom").strip() or "custom"
    if direct_solver_value not in ("custom", "cudss"):
        direct_solver_value = "custom"
    devsim.set_parameter(name="direct_solver", value=direct_solver_value)
    devsim.set_parameter(name="solver_callback", value=local_solver_callback)
    result_mode = os.environ.get("DEVSIM_CUDSS_RESULT_MODE", "").strip()
    if result_mode:
        devsim.set_parameter(name="cudss_result_mode", value=result_mode)


def _enable_baseline_solver_if_needed() -> None:
    import devsim

    try:
        current = devsim.get_parameter(name="direct_solver")
    except Exception:
        current = ""
    if current == "unknown":
        from umfpack.umfshim import local_solver_callback as umfpack_solver_callback

        devsim.set_parameter(name="direct_solver", value="custom")
        devsim.set_parameter(name="solver_callback", value=umfpack_solver_callback)


def _install_solve_timing_hook() -> dict[str, Any]:
    import devsim

    stats: dict[str, Any] = {
        "solve_calls": 0,
        "solve_wall_time": 0.0,
        "solve_failures": 0,
        "solve_args_samples": [],
    }
    original_solve = devsim.solve

    def _wrapped_solve(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        stats["solve_calls"] += 1
        if kwargs and len(stats["solve_args_samples"]) < 5:
            sample = {k: kwargs[k] for k in sorted(kwargs) if k in ("type", "absolute_error", "relative_error")}
            if sample:
                stats["solve_args_samples"].append(sample)
        try:
            return original_solve(*args, **kwargs)
        except Exception:
            stats["solve_failures"] += 1
            raise
        finally:
            stats["solve_wall_time"] += time.perf_counter() - start

    devsim.solve = _wrapped_solve
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--solver-mode", choices=("baseline", "cudss"), required=True)
    parser.add_argument("--devsim-so", default="")
    parser.add_argument("--arg", action="append", default=[])
    parser.add_argument("--timing-json", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))

    _load_devsim_module(args.devsim_so or None)
    if args.solver_mode == "cudss":
        _enable_cudss_solver()
    else:
        _enable_baseline_solver_if_needed()
    solve_stats = _install_solve_timing_hook()
    get_custom_cudss_stats = None
    if args.solver_mode == "cudss":
        direct_solver_value = os.environ.get("DEVSIM_CUDSS_DIRECT_SOLVER", "custom").strip() or "custom"
        if direct_solver_value == "custom":
            try:
                from cudss.cudss_shim import get_last_stats as _get_last_stats

                get_custom_cudss_stats = _get_last_stats
            except Exception:
                get_custom_cudss_stats = None

    start_total = time.perf_counter()
    os.chdir(args.working_dir)
    if args.working_dir not in sys.path:
        sys.path.insert(0, args.working_dir)
    script_path = Path(args.script)
    if not script_path.is_absolute():
        script_path = Path(args.working_dir) / script_path
    sys.argv = [script_path.name, *args.arg]
    error: str | None = None
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        total_wall_time = time.perf_counter() - start_total
        if args.timing_json:
            payload = {
                "mode": args.solver_mode,
                "script": str(script_path),
                "working_dir": args.working_dir,
                "total_wall_time": total_wall_time,
                "solve_calls": solve_stats["solve_calls"],
                "solve_wall_time": solve_stats["solve_wall_time"],
                "solve_failures": solve_stats["solve_failures"],
                "solve_args_samples": solve_stats["solve_args_samples"],
                "error": error,
            }
            if get_custom_cudss_stats is not None:
                try:
                    custom_stats = get_custom_cudss_stats()
                    if isinstance(custom_stats, dict):
                        payload["custom_cudss_stats"] = custom_stats
                except Exception:
                    pass
            Path(args.timing_json).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
