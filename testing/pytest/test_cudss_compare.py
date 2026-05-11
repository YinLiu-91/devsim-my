from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import uuid

import pytest

from case_parser import CaseSpec


_FLOAT_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z_])")
_EQN_RE = re.compile(r"number of equations\s+(\d+)")
_KV_NUM_RE = re.compile(r"([a-zA-Z0-9_]+)=([-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?)")
_UNSUPPORTED_SOLVE_RE = re.compile(
    r"solve\s*\(\s*type\s*=\s*['\"](?:ac|noise|transient[^'\"]*)",
    re.IGNORECASE | re.MULTILINE,
)
_COMPARE_IGNORE_PREFIXES = (
    "number of equations ",
    "Iteration:",
    "  Device:",
    "    Region:",
    "      Equation:",
    "  Circuit:",
)
_COMPARE_IGNORE_EXACT_PREFIXES = (
    "{'converged':",
    "{'absolute_error':",
)


def _filter_compare_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if not line.startswith(_COMPARE_IGNORE_PREFIXES)
        and not line.startswith(_COMPARE_IGNORE_EXACT_PREFIXES)
    ]


def _split_text_and_numbers(line: str) -> tuple[list[str], list[float]]:
    text_parts: list[str] = []
    numeric_parts: list[float] = []
    last = 0
    for m in _FLOAT_RE.finditer(line):
        text_parts.append(line[last : m.start()])
        numeric_parts.append(float(m.group(0)))
        last = m.end()
    text_parts.append(line[last:])
    return text_parts, numeric_parts


def _outputs_match_with_float_tol(
    baseline_text: str,
    cudss_text: str,
    abs_tol: float = 2e-7,
    rel_tol: float = 1e-8,
) -> bool:
    baseline_lines = _filter_compare_lines(baseline_text)
    cudss_lines = _filter_compare_lines(cudss_text)
    if len(baseline_lines) != len(cudss_lines):
        return False

    for bline, cline in zip(baseline_lines, cudss_lines):
        btext, bnums = _split_text_and_numbers(bline)
        ctext, cnums = _split_text_and_numbers(cline)
        if btext != ctext or len(bnums) != len(cnums):
            return False
        for bval, cval in zip(bnums, cnums):
            if not math.isclose(bval, cval, rel_tol=rel_tol, abs_tol=abs_tol):
                return False
    return True


def _run_case(
    case: CaseSpec, mode: str, devsim_so: str, timing_dir: Path | None = None
) -> tuple[subprocess.CompletedProcess[str], dict[str, object] | None]:
    cmd = [
        sys.executable,
        "/devsim/testing/pytest/run_case.py",
        "--script",
        case.script,
        "--working-dir",
        case.working_dir,
        "--solver-mode",
        mode,
    ]
    if devsim_so:
        cmd.extend(["--devsim-so", devsim_so])
    timing_file: Path | None = None
    if timing_dir is not None:
        timing_file = timing_dir / f"timing_{mode}_{uuid.uuid4().hex}.json"
        cmd.extend(["--timing-json", str(timing_file)])
    for a in case.script_args:
        cmd.extend(["--arg", a])
    cp = subprocess.run(cmd, capture_output=True, text=True)
    timing_payload: dict[str, object] | None = None
    if timing_file and timing_file.exists():
        timing_payload = json.loads(timing_file.read_text(encoding="utf-8"))
        timing_file.unlink()
    return cp, timing_payload


def _supports_phase1_cudss(case: CaseSpec) -> bool:
    script_path = Path(case.script)
    if not script_path.is_absolute():
        script_path = Path(case.working_dir) / script_path
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    return _UNSUPPORTED_SOLVE_RE.search(text) is None


def _run_prerequisites(
    case: CaseSpec, case_index: dict[str, CaseSpec], devsim_so: str, visited: set[str]
) -> None:
    for dep_name in case.depends:
        if dep_name in visited:
            continue
        dep_case = case_index.get(dep_name)
        if dep_case is None:
            continue
        _run_prerequisites(dep_case, case_index, devsim_so, visited)
        dep_result, _ = _run_case(dep_case, "baseline", devsim_so)
        assert dep_result.returncode == 0, (
            f"dependency baseline failed for {case.name} <- {dep_name}\n"
            f"stdout:\n{dep_result.stdout}\n\nstderr:\n{dep_result.stderr}"
        )
        visited.add(dep_name)


def _to_float(data: dict[str, object] | None, key: str) -> float:
    if not data:
        return 0.0
    val = data.get(key, 0.0)
    return float(val) if isinstance(val, (int, float)) else 0.0


def _to_int(data: dict[str, object] | None, key: str) -> int:
    if not data:
        return 0
    val = data.get(key, 0)
    return int(val) if isinstance(val, (int, float)) else 0


def _parse_max_equations(text: str) -> int:
    vals = [int(m.group(1)) for m in _EQN_RE.finditer(text)]
    return max(vals) if vals else 0


def _parse_cudss_transfer(text: str) -> dict[str, int | float]:
    transfer_lines = [line for line in text.splitlines() if "cuDSS transfer stats:" in line]
    if not transfer_lines:
        return {}
    kvs = dict(_KV_NUM_RE.findall(transfer_lines[-1]))
    out: dict[str, int | float] = {}
    remap = {
        "solve_calls": "shim_solve_calls",
    }
    float_fields = {
        "analysis_seconds",
        "factorization_seconds",
        "refactor_seconds",
        "factor_total_seconds",
        "solve_total_seconds",
        "solve_h2d_seconds",
        "solve_execute_seconds",
        "solve_d2h_seconds",
    }
    for key, value in kvs.items():
        dst = remap.get(key, key)
        if dst in float_fields:
            out[dst] = float(value)
        else:
            out[dst] = int(float(value))
    return out


def _parse_cudss_profile(text: str) -> dict[str, int | float]:
    profile_lines = [line for line in text.splitlines() if "cuDSS profile:" in line]
    if not profile_lines:
        return {}
    kvs = dict(_KV_NUM_RE.findall(profile_lines[-1]))

    int_fields = {
        "host_materialize_mode_calls",
        "host_materialize_node_calls",
        "analysis_miss_first_factor_calls",
        "analysis_miss_symbolic_status_calls",
        "analysis_miss_hash_mismatch_calls",
        "analysis_miss_dim_change_calls",
        "analysis_miss_backend_mode_calls",
    }
    float_fields = {
        "analysis_seconds",
        "refactor_seconds",
        "factor_seconds",
        "solve_seconds",
    }
    out: dict[str, int | float] = {}
    for key in int_fields:
        if key in kvs:
            out[key] = int(float(kvs[key]))
    for key in float_fields:
        if key in kvs:
            target_key = "factor_seconds_profile" if key == "factor_seconds" else ("solve_seconds_profile" if key == "solve_seconds" else key)
            out[target_key] = float(kvs[key])
    return out


def _parse_cudss_external_profile(text: str) -> dict[str, float]:
    lines = [line for line in text.splitlines() if "cuDSS external profile:" in line]
    if not lines:
        return {}
    kvs = dict(_KV_NUM_RE.findall(lines[-1]))
    float_fields = (
        "factor_total_seconds",
        "factor_build_seconds",
        "factor_interpreter_seconds",
        "factor_result_seconds",
        "factor_validate_seconds",
        "factor_parse_seconds",
        "solve_total_seconds",
        "solve_build_seconds",
        "solve_interpreter_seconds",
        "solve_result_seconds",
        "solve_validate_seconds",
        "solve_device_meta_seconds",
        "solve_parse_seconds",
        "solve_materialize_seconds",
        "solve_device_view_seconds",
    )
    out: dict[str, float] = {}
    for key in float_fields:
        if key in kvs:
            out[f"external_{key}"] = float(kvs[key])
    return out


def _parse_numeric_lines(text: str, marker: str, int_fields: set[str] | None = None, prefix: str = "") -> list[dict[str, int | float]]:
    lines = [line for line in text.splitlines() if marker in line]
    out: list[dict[str, int | float]] = []
    int_fields = int_fields or set()
    for line in lines:
        kvs = dict(_KV_NUM_RE.findall(line))
        row: dict[str, int | float] = {}
        for key, value in kvs.items():
            target = f"{prefix}{key}" if prefix else key
            if key in int_fields:
                row[target] = int(float(value))
            else:
                row[target] = float(value)
        out.append(row)
    return out


def _parse_cudss_solver_profile(text: str) -> dict[str, int | float]:
    rows = _parse_numeric_lines(text, "cuDSS solver profile:", {"iterations"}, prefix="solver_")
    return rows[-1] if rows else {}


def _parse_cudss_solver_iterations(text: str) -> list[dict[str, int | float]]:
    return _parse_numeric_lines(text, "cuDSS solver iteration:", {"iter"}, prefix="solver_")


def _parse_cudss_external_factor_calls(text: str) -> list[dict[str, int | float]]:
    return _parse_numeric_lines(text, "cuDSS external factor call:", {"call"}, prefix="external_factor_")


def _parse_cudss_external_solve_calls(text: str) -> list[dict[str, int | float]]:
    return _parse_numeric_lines(
        text,
        "cuDSS external solve call:",
        {"call", "need_host_vector", "device_view_enabled"},
        prefix="external_solve_",
    )


def _emit_timing(
    case: CaseSpec,
    baseline_timing: dict[str, object] | None,
    cudss_timing: dict[str, object] | None,
    baseline_proc: subprocess.CompletedProcess[str],
    cudss_proc: subprocess.CompletedProcess[str],
    pytestconfig: pytest.Config,
    print_timing: bool,
    timing_json: str,
) -> None:
    b_total = _to_float(baseline_timing, "total_wall_time")
    c_total = _to_float(cudss_timing, "total_wall_time")
    b_solve = _to_float(baseline_timing, "solve_wall_time")
    c_solve = _to_float(cudss_timing, "solve_wall_time")
    b_calls = _to_int(baseline_timing, "solve_calls")
    c_calls = _to_int(cudss_timing, "solve_calls")
    speedup = (b_total / c_total) if c_total > 0 else float("inf")
    solve_only_speedup = (b_solve / c_solve) if c_solve > 0 else float("inf")
    solve_share_baseline = (b_solve / b_total) if b_total > 0 else 0.0
    solve_share_cudss = (c_solve / c_total) if c_total > 0 else 0.0
    delta = b_total - c_total
    solver_delta = b_solve - c_solve
    nonsolver_b = max(0.0, b_total - b_solve)
    nonsolver_c = max(0.0, c_total - c_solve)
    eqn_count = max(_parse_max_equations(baseline_proc.stdout), _parse_max_equations(cudss_proc.stdout))
    baseline_solver_profile = _parse_cudss_solver_profile(baseline_proc.stdout)
    cudss_transfer = _parse_cudss_transfer(cudss_proc.stdout)
    cudss_profile = _parse_cudss_profile(cudss_proc.stdout)
    cudss_external = _parse_cudss_external_profile(cudss_proc.stdout)
    cudss_solver_profile = _parse_cudss_solver_profile(cudss_proc.stdout)
    cudss_solver_iterations = _parse_cudss_solver_iterations(cudss_proc.stdout)
    cudss_external_factor_calls = _parse_cudss_external_factor_calls(cudss_proc.stdout)
    cudss_external_solve_calls = _parse_cudss_external_solve_calls(cudss_proc.stdout)

    report = {
        "case": case.name,
        "baseline_total_s": b_total,
        "cudss_total_s": c_total,
        "speedup": speedup,
        "solve_only_speedup": solve_only_speedup,
        "delta_s": delta,
        "baseline_solver_s": b_solve,
        "cudss_solver_s": c_solve,
        "solve_share_baseline": solve_share_baseline,
        "solve_share_cudss": solve_share_cudss,
        "solver_delta_s": solver_delta,
        "baseline_nonsolver_s": nonsolver_b,
        "cudss_nonsolver_s": nonsolver_c,
        "baseline_solve_calls": b_calls,
        "cudss_solve_calls": c_calls,
        "equation_count": eqn_count,
    }
    if "solver_iterations" in baseline_solver_profile:
        report["baseline_newton_iterations"] = int(baseline_solver_profile["solver_iterations"])
    if "solver_iterations" in cudss_solver_profile:
        report["cudss_newton_iterations"] = int(cudss_solver_profile["solver_iterations"])
    report.update(cudss_transfer)
    report.update(cudss_profile)
    report.update(cudss_external)
    report.update(cudss_solver_profile)
    if cudss_solver_iterations:
        report["solver_iteration_breakdown"] = cudss_solver_iterations
    if cudss_external_factor_calls:
        report["external_factor_call_breakdown"] = cudss_external_factor_calls
    if cudss_external_solve_calls:
        report["external_solve_call_breakdown"] = cudss_external_solve_calls
    if "solver_total_seconds" in report:
        solver_total_internal = float(report["solver_total_seconds"])
        solver_linear_total = float(report.get("solver_iteration_linear_solve_seconds", 0.0))
        report["solver_framework_seconds"] = max(0.0, solver_total_internal - solver_linear_total)
        report["solver_python_wrap_seconds"] = max(0.0, c_solve - solver_total_internal)
    if "external_solve_total_seconds" in report:
        ext_total = float(report["external_solve_total_seconds"])
        ext_known = (
            float(report.get("external_solve_build_seconds", 0.0))
            + float(report.get("external_solve_interpreter_seconds", 0.0))
            + float(report.get("external_solve_result_seconds", 0.0))
            + float(report.get("external_solve_validate_seconds", 0.0))
            + float(report.get("external_solve_device_meta_seconds", 0.0))
            + float(report.get("external_solve_materialize_seconds", 0.0))
            + float(report.get("external_solve_device_view_seconds", 0.0))
        )
        report["external_solve_other_seconds"] = max(0.0, ext_total - ext_known)
    if "external_factor_total_seconds" in report:
        ext_factor_total = float(report["external_factor_total_seconds"])
        ext_factor_known = (
            float(report.get("external_factor_build_seconds", 0.0))
            + float(report.get("external_factor_interpreter_seconds", 0.0))
            + float(report.get("external_factor_result_seconds", 0.0))
            + float(report.get("external_factor_validate_seconds", 0.0))
        )
        report["external_factor_other_seconds"] = max(0.0, ext_factor_total - ext_factor_known)
    custom_stats = cudss_timing.get("custom_cudss_stats") if isinstance(cudss_timing, dict) else None
    if isinstance(custom_stats, dict):
        for key in (
            "analysis_seconds",
            "factorization_seconds",
            "refactor_seconds",
            "factor_total_seconds",
            "solve_total_seconds",
            "solve_h2d_seconds",
            "solve_execute_seconds",
            "solve_d2h_seconds",
            "solve_call_breakdown",
        ):
            if key in custom_stats:
                report[f"custom_{key}"] = custom_stats[key]
        if isinstance(custom_stats.get("solve_total_seconds"), (int, float)):
            csolve_total = float(custom_stats.get("solve_total_seconds", 0.0))
            csolve_h2d = float(custom_stats.get("solve_h2d_seconds", 0.0))
            csolve_exec = float(custom_stats.get("solve_execute_seconds", 0.0))
            csolve_d2h = float(custom_stats.get("solve_d2h_seconds", 0.0))
            report["custom_solve_other_seconds"] = max(0.0, csolve_total - csolve_h2d - csolve_exec - csolve_d2h)
            report["custom_solver_other_seconds"] = max(
                0.0,
                c_solve - float(custom_stats.get("factor_total_seconds", 0.0)) - csolve_total,
            )
    if all(
        key in report
        for key in (
            "external_factor_total_seconds",
            "external_solve_total_seconds",
            "solver_iteration_linear_solve_seconds",
        )
    ):
        linear_total = float(report["solver_iteration_linear_solve_seconds"])
        external_linear_total = float(report["external_factor_total_seconds"]) + float(report["external_solve_total_seconds"])
        report["linear_wrapper_other_seconds"] = max(0.0, linear_total - external_linear_total)
    if "external_factor_total_seconds" in report and "custom_factor_total_seconds" in report:
        report["external_factor_overhead_seconds"] = max(
            0.0,
            float(report["external_factor_total_seconds"]) - float(report["custom_factor_total_seconds"]),
        )
    if "external_solve_total_seconds" in report and "custom_solve_total_seconds" in report:
        report["external_solve_overhead_seconds"] = max(
            0.0,
            float(report["external_solve_total_seconds"]) - float(report["custom_solve_total_seconds"]),
        )
    if cudss_profile:
        solver_wrapper_s = max(0.0, c_solve - float(cudss_profile.get("solve_seconds_profile", 0.0)))
        report["cudss_solver_wrapper_s"] = solver_wrapper_s

    if print_timing:
        tr = pytestconfig.pluginmanager.getplugin("terminalreporter")
        line = (
            f"[timing] {case.name}: baseline={b_total:.6f}s cudss={c_total:.6f}s "
            f"speedup={speedup:.3f}x delta={delta:.6f}s | "
            f"solver={b_solve:.6f}->{c_solve:.6f}s ({solver_delta:.6f}s), "
            f"solve_only_speedup={solve_only_speedup:.3f}x, "
            f"solve_share={solve_share_baseline:.3f}->{solve_share_cudss:.3f}, "
            f"nonsolver={nonsolver_b:.6f}->{nonsolver_c:.6f}s, solve_calls={b_calls}/{c_calls}, "
            f"eqns={eqn_count}"
        )
        if case.name == "testing/cap2":
            line += " [focus=solve_only]"
        if "baseline_newton_iterations" in report or "cudss_newton_iterations" in report:
            line += (
                f", newton_iters={int(report.get('baseline_newton_iterations', 0))}/"
                f"{int(report.get('cudss_newton_iterations', 0))}"
            )
        if cudss_transfer:
            line += (
                f", h2d={cudss_transfer['h2d_bytes']}, d2h={cudss_transfer['d2h_bytes']}, "
                f"analysis={cudss_transfer['analysis_calls']}, refactor={cudss_transfer['refactor_calls']}"
            )
            if "config_set_applied" in cudss_transfer:
                line += f", config_set_applied={cudss_transfer['config_set_applied']}"
            if "mt_mode" in cudss_transfer:
                line += f", mt_mode={cudss_transfer['mt_mode']}"
            if "stream_mode" in cudss_transfer:
                line += f", stream_mode={cudss_transfer['stream_mode']}"
            if all(k in cudss_transfer for k in ("analysis_seconds", "factorization_seconds", "refactor_seconds", "solve_total_seconds")):
                line += (
                    ", phase_s_custom=("
                    f"analysis:{float(cudss_transfer['analysis_seconds']):.6f},"
                    f"factorization:{float(cudss_transfer['factorization_seconds']):.6f},"
                    f"refactor:{float(cudss_transfer['refactor_seconds']):.6f},"
                    f"solve_total:{float(cudss_transfer['solve_total_seconds']):.6f})"
                )
            if all(k in cudss_transfer for k in ("solve_h2d_seconds", "solve_execute_seconds", "solve_d2h_seconds", "solve_total_seconds")):
                solve_other = max(
                    0.0,
                    float(cudss_transfer["solve_total_seconds"])
                    - float(cudss_transfer["solve_h2d_seconds"])
                    - float(cudss_transfer["solve_execute_seconds"])
                    - float(cudss_transfer["solve_d2h_seconds"]),
                )
                line += (
                    ", solve_subphase_s=("
                    f"h2d:{float(cudss_transfer['solve_h2d_seconds']):.6f},"
                    f"execute:{float(cudss_transfer['solve_execute_seconds']):.6f},"
                    f"d2h:{float(cudss_transfer['solve_d2h_seconds']):.6f},"
                    f"other:{solve_other:.6f})"
                )
        if cudss_profile:
            hotspot_name = "solve"
            hotspot_value = float(cudss_profile.get("solve_seconds_profile", 0.0))
            analysis_s = float(cudss_profile.get("analysis_seconds", 0.0))
            refactor_s = float(cudss_profile.get("refactor_seconds", 0.0))
            factor_s = float(cudss_profile.get("factor_seconds_profile", 0.0))
            solve_s = float(cudss_profile.get("solve_seconds_profile", 0.0))
            for name, value in (
                ("analysis", analysis_s),
                ("refactor", refactor_s),
                ("solve", solve_s),
            ):
                if value >= hotspot_value:
                    hotspot_name = name
                    hotspot_value = value
            line += (
                f", host_mode={int(cudss_profile.get('host_materialize_mode_calls', 0))}, "
                f"host_node={int(cudss_profile.get('host_materialize_node_calls', 0))}, "
                f"analysis_miss=("
                f"first:{int(cudss_profile.get('analysis_miss_first_factor_calls', 0))},"
                f"symbolic:{int(cudss_profile.get('analysis_miss_symbolic_status_calls', 0))},"
                f"hash:{int(cudss_profile.get('analysis_miss_hash_mismatch_calls', 0))},"
                f"dim:{int(cudss_profile.get('analysis_miss_dim_change_calls', 0))},"
                f"backend:{int(cudss_profile.get('analysis_miss_backend_mode_calls', 0))}), "
                f"phase_s=(analysis:{analysis_s:.6f},refactor:{refactor_s:.6f},"
                f"factor:{factor_s:.6f},solve:{solve_s:.6f}), "
                f"solver_wrap={max(0.0, c_solve - solve_s):.6f}s, "
                f"phase_hotspot={hotspot_name}({hotspot_value:.6f}s)"
            )
        if cudss_external:
            ext_total = float(cudss_external.get("external_solve_total_seconds", 0.0))
            ext_build = float(cudss_external.get("external_solve_build_seconds", 0.0))
            ext_interp = float(cudss_external.get("external_solve_interpreter_seconds", 0.0))
            ext_result = float(cudss_external.get("external_solve_result_seconds", 0.0))
            ext_validate = float(cudss_external.get("external_solve_validate_seconds", 0.0))
            ext_device_meta = float(cudss_external.get("external_solve_device_meta_seconds", 0.0))
            ext_mat = float(cudss_external.get("external_solve_materialize_seconds", 0.0))
            ext_view = float(cudss_external.get("external_solve_device_view_seconds", 0.0))
            ext_other = max(0.0, ext_total - ext_build - ext_interp - ext_result - ext_validate - ext_device_meta - ext_mat - ext_view)
            line += (
                ", external_solve_s=("
                f"build:{ext_build:.6f},interp:{ext_interp:.6f},result:{ext_result:.6f},"
                f"validate:{ext_validate:.6f},meta:{ext_device_meta:.6f},materialize:{ext_mat:.6f},"
                f"device_view:{ext_view:.6f},other:{ext_other:.6f},total:{ext_total:.6f})"
            )
        if cudss_solver_profile:
            solver_total_internal = float(cudss_solver_profile.get("solver_total_seconds", 0.0))
            solver_setup = float(cudss_solver_profile.get("solver_setup_total_seconds", 0.0))
            solver_iteration = float(cudss_solver_profile.get("solver_iteration_total_seconds", 0.0))
            solver_post = float(cudss_solver_profile.get("solver_post_total_seconds", 0.0))
            solver_linear = float(cudss_solver_profile.get("solver_iteration_linear_solve_seconds", 0.0))
            solver_wrap = max(0.0, c_solve - solver_total_internal)
            line += (
                ", solver_profile_s=("
                f"setup:{solver_setup:.6f},iteration:{solver_iteration:.6f},post:{solver_post:.6f},"
                f"linear:{solver_linear:.6f},python_wrap:{solver_wrap:.6f},total:{solver_total_internal:.6f})"
            )
        layer_keys = (
            "solver_framework_seconds",
            "linear_wrapper_other_seconds",
            "external_factor_overhead_seconds",
            "external_solve_overhead_seconds",
            "solver_python_wrap_seconds",
        )
        if all(key in report for key in layer_keys):
            line += (
                ", solver_other_layers_s=("
                f"framework:{float(report['solver_framework_seconds']):.6f},"
                f"linear_wrap:{float(report['linear_wrapper_other_seconds']):.6f},"
                f"external_factor:{float(report['external_factor_overhead_seconds']):.6f},"
                f"external_solve:{float(report['external_solve_overhead_seconds']):.6f},"
                f"python_wrap:{float(report['solver_python_wrap_seconds']):.6f})"
            )
        if tr:
            tr.write_line(line)
            if cudss_solver_iterations:
                for item in cudss_solver_iterations:
                    tr.write_line(
                        "[timing][solver-iter] "
                        f"{case.name}: iter={int(item.get('solver_iter', 0))} "
                        f"total={float(item.get('solver_total_seconds', 0.0)):.6f}s "
                        f"load_dc={float(item.get('solver_load_dc_seconds', 0.0)):.6f}s "
                        f"load_time={float(item.get('solver_load_time_seconds', 0.0)):.6f}s "
                        f"linear={float(item.get('solver_linear_solve_seconds', 0.0)):.6f}s "
                        f"device_update={float(item.get('solver_device_update_seconds', 0.0)):.6f}s "
                        f"error={float(item.get('solver_error_seconds', 0.0)):.6f}s "
                        f"other={float(item.get('solver_other_seconds', 0.0)):.6f}s"
                    )
            if cudss_external_factor_calls:
                for item in cudss_external_factor_calls:
                    tr.write_line(
                        "[timing][external-factor] "
                        f"{case.name}: call={int(item.get('external_factor_call', 0))} "
                        f"total={float(item.get('external_factor_total_seconds', 0.0)):.6f}s "
                        f"build={float(item.get('external_factor_build_seconds', 0.0)):.6f}s "
                        f"interp={float(item.get('external_factor_interpreter_seconds', 0.0)):.6f}s "
                        f"result={float(item.get('external_factor_result_seconds', 0.0)):.6f}s "
                        f"validate={float(item.get('external_factor_validate_seconds', 0.0)):.6f}s "
                        f"other={float(item.get('external_factor_other_seconds', 0.0)):.6f}s"
                    )
            if cudss_external_solve_calls:
                for item in cudss_external_solve_calls:
                    tr.write_line(
                        "[timing][external-solve] "
                        f"{case.name}: call={int(item.get('external_solve_call', 0))} "
                        f"total={float(item.get('external_solve_total_seconds', 0.0)):.6f}s "
                        f"build={float(item.get('external_solve_build_seconds', 0.0)):.6f}s "
                        f"interp={float(item.get('external_solve_interpreter_seconds', 0.0)):.6f}s "
                        f"result={float(item.get('external_solve_result_seconds', 0.0)):.6f}s "
                        f"validate={float(item.get('external_solve_validate_seconds', 0.0)):.6f}s "
                        f"meta={float(item.get('external_solve_device_meta_seconds', 0.0)):.6f}s "
                        f"materialize={float(item.get('external_solve_materialize_seconds', 0.0)):.6f}s "
                        f"device_view={float(item.get('external_solve_device_view_seconds', 0.0)):.6f}s "
                        f"other={float(item.get('external_solve_other_seconds', 0.0)):.6f}s"
                    )
            if isinstance(report.get("custom_solve_call_breakdown"), list):
                for item in report["custom_solve_call_breakdown"]:
                    tr.write_line(
                        "[timing][shim-solve] "
                        f"{case.name}: call={int(float(item.get('call', 0.0)))} "
                        f"total={float(item.get('total_seconds', 0.0)):.6f}s "
                        f"h2d={float(item.get('h2d_seconds', 0.0)):.6f}s "
                        f"execute={float(item.get('execute_seconds', 0.0)):.6f}s "
                        f"d2h={float(item.get('d2h_seconds', 0.0)):.6f}s "
                        f"other={float(item.get('other_seconds', 0.0)):.6f}s"
                    )
        else:
            print(line)

    if timing_json:
        out = Path(timing_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(report, sort_keys=True) + "\n")


def test_case_baseline_vs_cudss(
    case_spec: CaseSpec,
    solver_mode: str,
    devsim_so: str,
    cudss_available: bool,
    strict_cudss: bool,
    case_index: dict[str, CaseSpec],
    pytestconfig: pytest.Config,
    print_timing: bool,
    timing_json: str,
) -> None:
    if solver_mode == "cudss" and not cudss_available:
        pytest.skip("cuDSS runtime/GPU unavailable")

    if solver_mode in ("baseline", "both"):
        _run_prerequisites(case_spec, case_index, devsim_so, set())
        baseline, baseline_timing = _run_case(case_spec, "baseline", devsim_so, Path(case_spec.working_dir))
        assert baseline.returncode == 0, (
            f"baseline failed for {case_spec.name}\n"
            f"stdout:\n{baseline.stdout}\n\nstderr:\n{baseline.stderr}"
        )
    else:
        baseline = None
        baseline_timing = None

    if solver_mode in ("cudss", "both"):
        if not cudss_available:
            pytest.skip("cuDSS runtime/GPU unavailable; baseline already executed")
        if not _supports_phase1_cudss(case_spec):
            pytest.skip("Phase-1 cuDSS supports DC real path only")
        _run_prerequisites(case_spec, case_index, devsim_so, set())
        cudss, cudss_timing = _run_case(case_spec, "cudss", devsim_so, Path(case_spec.working_dir))
        if cudss.returncode != 0:
            msg = (
                f"cudss failed for {case_spec.name}\n"
                f"stdout:\n{cudss.stdout}\n\nstderr:\n{cudss.stderr}"
            )
            if strict_cudss:
                assert cudss.returncode == 0, msg
            pytest.xfail(msg)
    else:
        cudss = None
        cudss_timing = None

    if solver_mode == "both":
        assert baseline is not None and cudss is not None
        _emit_timing(case_spec, baseline_timing, cudss_timing, baseline, cudss, pytestconfig, print_timing, timing_json)
        if not _outputs_match_with_float_tol(baseline.stdout, cudss.stdout):
            msg = (
                f"baseline/cudss output mismatch for {case_spec.name}\n"
                f"case={asdict(case_spec)}\n"
            )
            if strict_cudss:
                assert _outputs_match_with_float_tol(baseline.stdout, cudss.stdout), msg
            pytest.xfail(msg)
