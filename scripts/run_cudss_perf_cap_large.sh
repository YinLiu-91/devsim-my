#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cudss_common.sh"
ROOT_DIR="${cudss_root_dir}"
cd "${ROOT_DIR}"

REPEATS="${REPEATS:-3}"
CAP2D_LARGE_MESH_SCALE="${CAP2D_LARGE_MESH_SCALE:-0.1}"
DEVSIM_SO="$(cudss_require_devsim_so)"

cudss_export_native_mt_defaults

export REPEATS CAP2D_LARGE_MESH_SCALE DEVSIM_SO

python3 - <<'PY'
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile

ROOT = pathlib.Path.cwd()
REPEATS = int(os.environ["REPEATS"])
MESH_SCALE = os.environ["CAP2D_LARGE_MESH_SCALE"]
DEVSIM_SO = os.environ["DEVSIM_SO"]
CUDSS_ENV = {
    key: os.environ[key]
    for key in (
        "DEVSIM_CUDSS_DIRECT_SOLVER",
        "DEVSIM_CUDSS_BACKEND_POLICY",
        "DEVSIM_CUDSS_MT_MODE",
        "DEVSIM_CUDSS_USE_STREAM",
        "DEVSIM_CUDSS_RESULT_MODE",
    )
}

if not pathlib.Path(DEVSIM_SO).is_file():
    raise SystemExit(f"DEVSIM_SO not found: {DEVSIM_SO}")

BASE_CMD = [
    "python3",
    "testing/pytest/run_case.py",
    "--script",
    "cap2d_large.py",
    "--working-dir",
    str(ROOT / "examples/capacitance"),
    "--devsim-so",
    DEVSIM_SO,
]

PROFILE_PREFIX = "cuDSS solver profile:"
PROFILE_KEYS = (
    "iteration_load_dc_seconds",
    "iteration_linear_solve_seconds",
    "iteration_device_update_seconds",
    "iteration_finalize_seconds",
    "iteration_clear_seconds",
)


def parse_profile_line(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def median_of(records: list[dict[str, float]], key: str) -> float:
    return statistics.median(record[key] for record in records)


def run_mode(label: str, solver_mode: str, extra_env: dict[str, str]) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for index in range(1, REPEATS + 1):
        with tempfile.TemporaryDirectory(prefix=f"cap2d-large-{label}-") as tmpdir:
            tmpdir_path = pathlib.Path(tmpdir)
            out_path = tmpdir_path / "stdout.txt"
            err_path = tmpdir_path / "stderr.txt"
            json_path = tmpdir_path / "timing.json"
            env = os.environ.copy()
            env.update(
                {
                    "CAP2D_LARGE_MESH_SCALE": MESH_SCALE,
                    "DEVSIM_CUDSS_PROFILE": "1",
                }
            )
            env.update(extra_env)
            cmd = [*BASE_CMD, "--solver-mode", solver_mode, "--timing-json", str(json_path)]
            with out_path.open("w", encoding="utf-8") as stdout, err_path.open("w", encoding="utf-8") as stderr:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            if proc.returncode != 0:
                sys.stderr.write(err_path.read_text(encoding="utf-8", errors="ignore"))
                raise SystemExit(f"{label} run {index} failed with exit code {proc.returncode}")

            timing = json.loads(json_path.read_text(encoding="utf-8"))
            profile: dict[str, float] | None = None
            for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith(PROFILE_PREFIX):
                    profile = parse_profile_line(line)
                    break
            if profile is None:
                raise SystemExit(f"{label} run {index} did not emit '{PROFILE_PREFIX}'")

            record = {
                "total": float(timing["total_wall_time"]),
                "solve": float(timing["solve_wall_time"]),
            }
            for key in PROFILE_KEYS:
                record[key] = float(profile.get(key, 0.0))
            records.append(record)
    return records


baseline_runs = run_mode("baseline", "baseline", {})
cudss_runs = run_mode(
    "cudss-native-mt",
    "cudss",
    CUDSS_ENV,
)

baseline = {key: median_of(baseline_runs, key) for key in baseline_runs[0]}
cudss = {key: median_of(cudss_runs, key) for key in cudss_runs[0]}

print("# cap2d_large baseline vs current native+MT")
print(f"# repeats={REPEATS}")
print(f"# CAP2D_LARGE_MESH_SCALE={MESH_SCALE}")
print(f"# DEVSIM_SO={DEVSIM_SO}")
print(
    "# current cudss env: "
    f"BACKEND_POLICY={CUDSS_ENV['DEVSIM_CUDSS_BACKEND_POLICY']}, "
    f"MT_MODE={CUDSS_ENV['DEVSIM_CUDSS_MT_MODE']}, "
    f"USE_STREAM={CUDSS_ENV['DEVSIM_CUDSS_USE_STREAM']}, "
    f"RESULT_MODE={CUDSS_ENV['DEVSIM_CUDSS_RESULT_MODE']}"
)
print()
print("| mode | total(s) | solve(s) | load_dc(s) | linear_solve(s) | device_update(s) | finalize(s) | clear(s) |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
print(
    f"| baseline | {baseline['total']:.6f} | {baseline['solve']:.6f} | "
    f"{baseline['iteration_load_dc_seconds']:.6f} | {baseline['iteration_linear_solve_seconds']:.6f} | "
    f"{baseline['iteration_device_update_seconds']:.6f} | {baseline['iteration_finalize_seconds']:.6f} | "
    f"{baseline['iteration_clear_seconds']:.6f} |"
)
print(
    f"| cudss native+MT | {cudss['total']:.6f} | {cudss['solve']:.6f} | "
    f"{cudss['iteration_load_dc_seconds']:.6f} | {cudss['iteration_linear_solve_seconds']:.6f} | "
    f"{cudss['iteration_device_update_seconds']:.6f} | {cudss['iteration_finalize_seconds']:.6f} | "
    f"{cudss['iteration_clear_seconds']:.6f} |"
)
print(
    f"| speedup (baseline/cudss) | {baseline['total'] / cudss['total']:.3f}x | "
    f"{baseline['solve'] / cudss['solve']:.3f}x | - | - | - | - | - |"
)
print()
for label, records in (("baseline", baseline_runs), ("cudss native+MT", cudss_runs)):
    print(f"# raw {label} runs")
    for idx, record in enumerate(records, 1):
        print(
            f"run={idx} total={record['total']:.6f} solve={record['solve']:.6f} "
            f"load_dc={record['iteration_load_dc_seconds']:.6f} "
            f"linear_solve={record['iteration_linear_solve_seconds']:.6f} "
            f"device_update={record['iteration_device_update_seconds']:.6f}"
        )
    print()
PY
