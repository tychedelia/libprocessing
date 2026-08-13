# Grid-accelerated neighbour density. `apply(NEIGHBOR, op=DENSITY)` counts each
# particle's falloff-weighted neighbours within `radius` using the spatial grid
# (O(N), not O(N^2)) and writes it to a `density` attribute; a tiny compute maps
# that to color. Divergence-free curl noise stirs the cloud into filaments, and
# the density coloring lights up the dense strands — all on the GPU.
from mewnala import *
from random import uniform
from math import cos, sin

N = 50000
BOX = 20.0
RADIUS = 1.5

p = None
g = None
tint = None


def setup():
    global p, g, tint
    size(900, 700)
    window_title(f"Neighbour density — {N:,}")
    mode_3d()

    p = create_particles(
        capacity=N,
        attributes=[
            Attribute.position(),
            Attribute.color(),
            Attribute("density", AttributeFormat.Float),
        ],
    )
    p.buffer("position").write([[uniform(-BOX, BOX) for _ in range(3)] for _ in range(N)])

    # cell_size >= radius so the 3x3x3 cell block covers every neighbour.
    cells = int(2.0 * BOX / RADIUS) + 2
    g = p.create_grid(min=[-BOX, -BOX, -BOX], cell_size=RADIUS, dims=[cells, cells, cells])

    tint = create_compute(load_shader("shaders/density_color.wesl"))


def draw():
    background(3, 4, 9)

    t = elapsed_time
    d = BOX * 2.4
    camera_position(cos(t * 0.08) * d, BOX * 0.5, sin(t * 0.08) * d)
    camera_look_at(0.0, 0.0, 0.0)

    # Stir into filaments with divergence-free curl noise, wrap at the box.
    p.apply(NOISE, scale=0.12, strength=0.06, time=t * 0.15, divergence_free=1)
    p.apply(BOUNDS_BOX, aabb_min=[-BOX] * 3, aabb_max=[BOX] * 3, mode=2)

    # Local density -> color. The grid is rebuilt inside apply(NEIGHBOR).
    p.apply(NEIGHBOR, grid=g, out="density", op=DENSITY, radius=RADIUS)
    tint.set(scale=1.0 / 30.0)
    p.apply(tint)

    particles(p)  # colored points


run()
