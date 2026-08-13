# Direct rasterization, connected lines: the particle `position` buffer is drawn
# straight as a LINE_STRIP — consecutive particles i, i+1, i+2, ... form one
# continuous polyline, no index buffer, no connectivity, just `draw(0..count)`
# with the line-strip primitive. `topology` only picks the primitive; the vertex
# order IS the connectivity.
#
# The particles trace a (3,2) torus knot whose phase drifts over time; drawn as
# a strip they read as a smooth glowing curve.
from mewnala import *
from math import cos, sin, tau

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

    particles(p, topology="line_strip")  # consecutive particles -> one polyline


run()
