# Copyright 2013 DEVSIM LLC
#
# SPDX-License-Identifier: Apache-2.0

import os

from devsim import (
    add_2d_contact,
    add_2d_mesh_line,
    add_2d_region,
    contact_equation,
    contact_node_model,
    create_2d_mesh,
    create_device,
    edge_from_node_model,
    edge_model,
    element_from_edge_model,
    equation,
    finalize_mesh,
    get_contact_charge,
    node_model,
    node_solution,
    set_parameter,
    solve,
    write_devices,
)

device = "MyDevice"
region = "MyRegion"

# Smaller scale => denser mesh (larger solve).
mesh_scale = float(os.getenv("CAP2D_LARGE_MESH_SCALE", "1.0"))
if mesh_scale <= 0.0:
    raise RuntimeError("CAP2D_LARGE_MESH_SCALE must be > 0")


def ps_scaled(ps: float) -> float:
    return max(1.0e-6, ps * mesh_scale)

xmin = -25.0
xmax = 25.0
ymin = 0.0
ymax = 50.0

create_2d_mesh(mesh=device)

# Refined Y mesh around both electrodes.
for pos, ps in (
    (ymin, 0.03),
    (0.03, 0.02),
    (0.06, 0.02),
    (0.1, 0.02),
    (0.13, 0.02),
    (0.16, 0.02),
    (0.2, 0.03),
    (0.3, 0.05),
    (0.4, 0.05),
    (0.5, 0.05),
    (0.6, 0.05),
    (0.7, 0.05),
    (0.8, 0.03),
    (0.84, 0.02),
    (0.87, 0.02),
    (0.9, 0.02),
    (0.94, 0.02),
    (0.97, 0.02),
    (1.0, 0.03),
    (ymax, 1.5),
):
    add_2d_mesh_line(mesh=device, dir="y", pos=pos, ps=ps_scaled(ps))

# Refined X mesh around center/top-plate overlap.
for pos, ps in (
    (xmin, 5.0),
    (-24.975, 4.0),
    (-4.0, 2.0),
    (-2.0, 0.10),
    (-1.0, 0.05),
    (0.0, 0.05),
    (1.0, 0.05),
    (2.0, 0.10),
    (4.0, 2.0),
    (24.975, 4.0),
    (xmax, 5.0),
):
    add_2d_mesh_line(mesh=device, dir="x", pos=pos, ps=ps_scaled(ps))

add_2d_region(
    mesh=device, material="gas", region="air", yl=ymin, yh=ymax, xl=xmin, xh=xmax
)
add_2d_region(mesh=device, material="metal", region="m1", yl=0.1, yh=0.2, xl=-24.975, xh=24.975)
add_2d_region(mesh=device, material="metal", region="m2", yl=0.8, yh=0.9, xl=-2.0, xh=2.0)

add_2d_contact(
    mesh=device, name="bot", region="air", yl=0.1, yh=0.2, xl=-24.975, xh=24.975, material="metal"
)
add_2d_contact(
    mesh=device, name="top", region="air", yl=0.8, yh=0.9, xl=-2.0, xh=2.0, material="metal"
)
finalize_mesh(mesh=device)
create_device(mesh=device, device=device)
region = "air"

set_parameter(device=device, region=region, name="Permittivity", value=3.9 * 8.85e-14)

node_solution(device=device, region=region, name="Potential")
edge_from_node_model(device=device, region=region, node_model="Potential")

edge_model(
    device=device,
    region=region,
    name="ElectricField",
    equation="(Potential@n0 - Potential@n1)*EdgeInverseLength",
)
edge_model(device=device, region=region, name="ElectricField:Potential@n0", equation="EdgeInverseLength")
edge_model(device=device, region=region, name="ElectricField:Potential@n1", equation="-EdgeInverseLength")

edge_model(device=device, region=region, name="DField", equation="Permittivity*ElectricField")
edge_model(
    device=device,
    region=region,
    name="DField:Potential@n0",
    equation="diff(Permittivity*ElectricField, Potential@n0)",
)
edge_model(device=device, region=region, name="DField:Potential@n1", equation="-DField:Potential@n0")

equation(
    device=device,
    region=region,
    name="PotentialEquation",
    variable_name="Potential",
    edge_model="DField",
    variable_update="default",
)

for c in ("top", "bot"):
    contact_node_model(device=device, contact=c, name=f"{c}_bc", equation=f"Potential - {c}_bias")
    contact_node_model(device=device, contact=c, name=f"{c}_bc:Potential", equation="1")
    contact_equation(
        device=device,
        contact=c,
        name="PotentialEquation",
        node_model=f"{c}_bc",
        edge_charge_model="DField",
    )

set_parameter(device=device, name="top_bias", value=1.0)
set_parameter(device=device, name="bot_bias", value=0.0)

edge_model(device=device, region="m1", name="ElectricField", equation="0")
edge_model(device=device, region="m2", name="ElectricField", equation="0")
node_model(device=device, region="m1", name="Potential", equation="bot_bias;")
node_model(device=device, region="m2", name="Potential", equation="top_bias;")

solve(
    type="dc",
    absolute_error=1.0,
    relative_error=1e-10,
    maximum_iterations=30,
    solver_type="direct",
)

element_from_edge_model(edge_model="ElectricField", device=device, region=region)
print(get_contact_charge(device=device, contact="top", equation="PotentialEquation"))
print(get_contact_charge(device=device, contact="bot", equation="PotentialEquation"))

write_devices(file="cap2d_large.msh", type="devsim")
write_devices(file="cap2d_large.dat", type="tecplot")
