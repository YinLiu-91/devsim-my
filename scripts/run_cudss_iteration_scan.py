#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


def _load_last_jsonl(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _median(records: list[dict[str, object]], key: str) -> float | None:
    values = [float(r[key]) for r in records if isinstance(r.get(key), (int, float))]
    if not values:
        return None
    return statistics.median(values)


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "-"


def _mode_env(mode: str) -> dict[str, str]:
    if mode == "custom":
        return {
            "DEVSIM_CUDSS_DIRECT_SOLVER": "custom",
            "DEVSIM_CUDSS_BACKEND_POLICY": "callback",
            "DEVSIM_CUDSS_MT_MODE": "0",
            "DEVSIM_CUDSS_USE_STREAM": "0",
        }
    if mode == "native":
        return {
            "DEVSIM_CUDSS_DIRECT_SOLVER": "cudss",
            "DEVSIM_CUDSS_BACKEND_POLICY": "native",
            "DEVSIM_CUDSS_MT_MODE": "0",
            "DEVSIM_CUDSS_USE_STREAM": "0",
        }
    if mode == "native_mt":
        return {
            "DEVSIM_CUDSS_DIRECT_SOLVER": "cudss",
            "DEVSIM_CUDSS_BACKEND_POLICY": "native",
            "DEVSIM_CUDSS_MT_MODE": "1",
            "DEVSIM_CUDSS_USE_STREAM": "0",
        }
    if mode == "native_mt_stream":
        return {
            "DEVSIM_CUDSS_DIRECT_SOLVER": "cudss",
            "DEVSIM_CUDSS_BACKEND_POLICY": "native",
            "DEVSIM_CUDSS_MT_MODE": "1",
            "DEVSIM_CUDSS_USE_STREAM": "1",
        }
    raise ValueError(f"unsupported mode: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep nonlinear bias points and report Newton iterations vs cuDSS speedup.")
    parser.add_argument(
        "--case-filter",
        default="examples/diode/diode_1d_cudss_bench",
        help="pytest case filter passed to test_cudss_compare.py",
    )
    parser.add_argument(
        "--biases",
        default="0.0,0.1,0.2,0.3,0.4,0.5",
        help="comma-separated top-bias sweep values",
    )
    parser.add_argument(
        "--carrier-scales",
        default="1.0",
        help="comma-separated carrier scaling factors applied before the final solve",
    )
    parser.add_argument(
        "--potential-offsets",
        default="0.0",
        help="comma-separated Potential offsets applied before the final solve",
    )
    parser.add_argument(
        "--modes",
        default="custom,native,native_mt,native_mt_stream",
        help="comma-separated run modes: custom,native,native_mt,native_mt_stream",
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per point (median reported)")
    parser.add_argument(
        "--ramp-steps",
        type=int,
        default=5,
        help="number of DC ramp steps before the final bias point",
    )
    parser.add_argument(
        "--devsim-so",
        default="/devsim/linux_x86_64_release/src/main/devsim_py3.so",
        help="devsim_py3 shared object to load during pytest runs",
    )
    args = parser.parse_args()

    biases = [float(x) for x in args.biases.split(",") if x.strip()]
    carrier_scales = [float(x) for x in args.carrier_scales.split(",") if x.strip()]
    potential_offsets = [float(x) for x in args.potential_offsets.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    base_env = {
        "DEVSIM_CUDSS_PROFILE": "1",
        "DEVSIM_CUDSS_RESULT_MODE": os.environ.get("DEVSIM_CUDSS_RESULT_MODE", "device_experimental"),
        "DEVSIM_CUDSS_PINNED_STAGING": os.environ.get("DEVSIM_CUDSS_PINNED_STAGING", "1"),
    }

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="cudss_iter_scan_") as td:
        timing_json = Path(td) / "timing.jsonl"
        for mode in modes:
            for bias in biases:
                for carrier_scale in carrier_scales:
                    for potential_offset in potential_offsets:
                        env = os.environ.copy()
                        env.update(base_env)
                        env.update(_mode_env(mode))
                        env["DIODE_1D_BENCH_BIAS"] = f"{bias:.6f}"
                        env["DIODE_1D_BENCH_CARRIER_SCALE"] = f"{carrier_scale:.6f}"
                        env["DIODE_1D_BENCH_POTENTIAL_OFFSET"] = f"{potential_offset:.6f}"
                        env["DIODE_1D_BENCH_RAMP_STEPS"] = str(max(0, args.ramp_steps))

                        run_records: list[dict[str, object]] = []
                        run_rcs: list[int] = []
                        errors: list[str] = []
                        for _ in range(max(1, args.repeats)):
                            if timing_json.exists():
                                timing_json.unlink()
                            cmd = [
                                "python3",
                                "-m",
                                "pytest",
                                "testing/pytest/test_cudss_compare.py",
                                f"--case-filter={args.case_filter}",
                                "--solver-mode=both",
                                "--print-timing",
                                f"--timing-json={timing_json}",
                                "-q",
                            ]
                            if args.devsim_so:
                                cmd.extend(["--devsim-so", args.devsim_so])
                            cp = subprocess.run(cmd, env=env, text=True, capture_output=True, cwd="/devsim")
                            run_rcs.append(cp.returncode)
                            record = _load_last_jsonl(timing_json)
                            if record:
                                run_records.append(record)
                            else:
                                errors.append((cp.stderr or cp.stdout).strip() or "no timing record")

                        row: dict[str, object] = {
                            "mode": mode,
                            "bias": bias,
                            "carrier_scale": carrier_scale,
                            "potential_offset": potential_offset,
                            "rc": max(run_rcs) if run_rcs else 1,
                        }
                        if run_records:
                            for key in (
                                "baseline_total_s",
                                "cudss_total_s",
                                "baseline_solver_s",
                                "cudss_solver_s",
                                "speedup",
                                "solve_only_speedup",
                                "baseline_newton_iterations",
                                "cudss_newton_iterations",
                                "mt_mode",
                                "stream_mode",
                                "equation_count",
                            ):
                                value = _median(run_records, key)
                                if value is not None:
                                    row[key] = value
                            row["samples"] = len(run_records)
                        else:
                            row["samples"] = 0
                            row["error"] = errors[-1] if errors else "no timing record"
                        rows.append(row)

    print(f"case-filter: {args.case_filter}")
    print("| mode | bias | carrier_scale | potential_offset | rc | samples | eqns | baseline_iter | cudss_iter | total_speedup | solve_only_speedup | baseline_solver_s | cudss_solver_s | baseline_total_s | cudss_total_s | mt_mode | stream_mode | note |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in sorted(rows, key=lambda item: (str(item.get("mode", "")), float(item.get("bias", 0.0)), float(item.get("carrier_scale", 1.0)), float(item.get("potential_offset", 0.0)))):
        print(
            "| {mode} | {bias} | {carrier_scale} | {potential_offset} | {rc} | {samples} | {eqns} | {baseline_iter} | {cudss_iter} | {speedup} | {solve_only} | "
            "{baseline_solver_s} | {cudss_solver_s} | {baseline_total_s} | {cudss_total_s} | {mt_mode} | {stream_mode} | {note} |".format(
                mode=row.get("mode", "-"),
                bias=_fmt(row.get("bias")),
                carrier_scale=_fmt(row.get("carrier_scale")),
                potential_offset=_fmt(row.get("potential_offset")),
                rc=row.get("rc", "-"),
                samples=row.get("samples", 0),
                eqns=_fmt(row.get("equation_count")),
                baseline_iter=_fmt(row.get("baseline_newton_iterations")),
                cudss_iter=_fmt(row.get("cudss_newton_iterations")),
                speedup=_fmt(row.get("speedup")),
                solve_only=_fmt(row.get("solve_only_speedup")),
                baseline_solver_s=_fmt(row.get("baseline_solver_s")),
                cudss_solver_s=_fmt(row.get("cudss_solver_s")),
                baseline_total_s=_fmt(row.get("baseline_total_s")),
                cudss_total_s=_fmt(row.get("cudss_total_s")),
                mt_mode=_fmt(row.get("mt_mode")),
                stream_mode=_fmt(row.get("stream_mode")),
                note=str(row.get("error", "")),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
