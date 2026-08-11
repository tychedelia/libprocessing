# Direct-rasterization smoke test: draw a particle `position` buffer straight
# as a 1px point cloud, with no instanced geometry. `particles(p)` (no shape
# argument) takes the custom point-cloud render pipeline — positions are pulled
# from the storage buffer by vertex index, projected by the camera, one point
# each. This is rung 1 of the direct-raster path (no connectivity, no indirect).
from mewnala import *
from math import cos, sin
from random import uniform

COUNT = 20000
BOUND = 20.0

p = None


def setup():
    global p
    size(900, 700)
    window_title(f"Particle points — {COUNT:,}")
    mode_3d()

    p = create_particles(
        capacity=COUNT,
        attributes=[Attribute.position()],
    )

    positions = [[uniform(-BOUND, BOUND) for _ in range(3)] for _ in range(COUNT)]
    p.buffer("position").write(positions)


def draw():
    background(6, 8, 14)

    t = elapsed_time * 0.15
    r = BOUND * 2.6
    camera_position(cos(t) * r, BOUND * 0.6, sin(t) * r)
    camera_look_at(0.0, 0.0, 0.0)

    particles(p)  # no geometry -> direct point-cloud raster


run()
