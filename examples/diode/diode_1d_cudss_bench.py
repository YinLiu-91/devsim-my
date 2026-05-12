import os

from devsim import get_node_model_values, set_node_values, set_parameter, solve

import diode_common
import python_packages.simple_physics as simple_physics


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


device = "MyDevice"
region = "MyRegion"

diode_common.CreateMesh(device=device, region=region)
diode_common.SetParameters(device=device, region=region)
set_parameter(device=device, region=region, name="taun", value=1e-8)
set_parameter(device=device, region=region, name="taup", value=1e-8)
diode_common.SetNetDoping(device=device, region=region)

diode_common.InitialSolution(device, region)
solve(
    type="dc",
    absolute_error=_get_env_float("DIODE_1D_BENCH_POISSON_ABSERR", 1.0),
    relative_error=_get_env_float("DIODE_1D_BENCH_POISSON_RELERR", 1e-10),
    maximum_iterations=_get_env_int("DIODE_1D_BENCH_POISSON_MAXITER", 30),
)

diode_common.DriftDiffusionInitialSolution(device, region)
solve(
    type="dc",
    absolute_error=_get_env_float("DIODE_1D_BENCH_EQUIL_ABSERR", 1e10),
    relative_error=_get_env_float("DIODE_1D_BENCH_EQUIL_RELERR", 1e-10),
    maximum_iterations=_get_env_int("DIODE_1D_BENCH_EQUIL_MAXITER", 30),
)

top_bias = _get_env_float("DIODE_1D_BENCH_BIAS", 0.0)
bot_bias = _get_env_float("DIODE_1D_BENCH_BOT_BIAS", 0.0)
carrier_scale = _get_env_float("DIODE_1D_BENCH_CARRIER_SCALE", 1.0)
if carrier_scale != 1.0:
    electrons = [v * carrier_scale for v in get_node_model_values(device=device, region=region, name="Electrons")]
    holes = [v * carrier_scale for v in get_node_model_values(device=device, region=region, name="Holes")]
    set_node_values(device=device, region=region, name="Electrons", values=electrons)
    set_node_values(device=device, region=region, name="Holes", values=holes)
potential_offset = _get_env_float("DIODE_1D_BENCH_POTENTIAL_OFFSET", 0.0)
if potential_offset != 0.0:
    potential = [
        v + potential_offset
        for v in get_node_model_values(device=device, region=region, name="Potential")
    ]
    set_node_values(device=device, region=region, name="Potential", values=potential)
ramp_steps = max(0, _get_env_int("DIODE_1D_BENCH_RAMP_STEPS", 0))
for step in range(1, max(1, ramp_steps) + 1):
    scale = 1.0 if ramp_steps == 0 else (step / ramp_steps)
    set_parameter(
        device=device,
        name=simple_physics.GetContactBiasName("top"),
        value=top_bias * scale,
    )
    set_parameter(
        device=device,
        name=simple_physics.GetContactBiasName("bot"),
        value=bot_bias * scale,
    )
    solve(
        type="dc",
        absolute_error=_get_env_float("DIODE_1D_BENCH_FINAL_ABSERR", 1e10),
        relative_error=_get_env_float("DIODE_1D_BENCH_FINAL_RELERR", 1e-10),
        maximum_iterations=_get_env_int("DIODE_1D_BENCH_FINAL_MAXITER", 30),
    )

print(
    f"diode_1d_cudss_bench bias={top_bias:.6f} bot_bias={bot_bias:.6f} "
    f"ramp_steps={ramp_steps} carrier_scale={carrier_scale:.6f} "
    f"potential_offset={potential_offset:.6f}"
)
simple_physics.PrintCurrents(device, "top")
simple_physics.PrintCurrents(device, "bot")
