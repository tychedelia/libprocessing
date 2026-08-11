# Direct-rasterization rung 2: draw a particle `position` buffer as connected
# LINES. A compute kernel generates the hardware index buffer AND the indirect
# draw args on the GPU (see connectivity.wgsl / point_render.rs), so the whole
# polyline is issued as a single `draw_indexed_indirect` — no CPU index list,
# no per-segment instancing. `particles(p, topology="lines")` chains consecutive
# particles i -> i+1.
#
# The particles trace a (3,2) torus knot whose phase drifts over time; the chain
# through them becomes a smooth glowing curve.
from mewnala import *
from math import cos, sin, pi, tau

N = 3000
P, Q = 3, 2  # torus knot winding
SCALE = 8.0

p = None


def knot(t, phase):
    # (P,Q) torus knot, radius modulated so it reads as a 3D curve.
    r = cos(Q * t) + 2.0
    x = r * cos(P * t + phase)
    y = r * sin(P * t + phase)
    z = -sin(Q * t)
    return (x * SCALE, y * SCALE, z * SCALE)


def setup():
    global p
    size(900, 700)
    window_title(f"Particle lines — {N:,}-point torus knot")
    mode_3d()

    p = create_particles(
        capacity=N,
        attributes=[Attribute.position()],
    )
    # Seed something so the buffer exists; draw() rewrites it each frame.
    p.buffer("position").write([list(knot(i / N * tau, 0.0)) for i in range(N)])


def draw():
    background(6, 8, 14)

    phase = elapsed_time * 0.4
    positions = [list(knot(i / N * tau, phase)) for i in range(N)]
    p.buffer("position").write(positions)

    t = elapsed_time * 0.12
    r = SCALE * 4.5
    camera_position(cos(t) * r, SCALE * 1.5, sin(t) * r)
    camera_look_at(0.0, 0.0, 0.0)

    particles(p, topology="lines")  # GPU-generated index buffer + indirect draw


run()
