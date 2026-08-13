# Direct rasterization, connected triangles: the particle `position` buffer is
# drawn as a TRIANGLE_STRIP — consecutive particles v0,v1,v2,v3,... form a strip
# of triangles (v0v1v2, v1v2v3, ...), no index buffer, just `draw(0..count)` with
# the triangle-strip primitive. Triangle topologies get flat per-face lighting.
#
# We lay the particles out as two interleaved rails of a twisting band (a Mobius
# strip): [left0, right0, left1, right1, ...]. The vertex order IS the surface —
# that is all "connectivity" means here. Arrange the particles differently (in a
# compute shader, say) and the same call rasterizes whatever surface they form.
from mewnala import *
from math import cos, sin, tau

SEGMENTS = 400            # samples around the band
N = SEGMENTS * 2          # two rail vertices (left/right) per sample
R = 10.0                  # band radius
WIDTH = 3.5               # half-width of the band

p = None


def band(seg, phase):
    # Center on a circle in the xz-plane; the half-width vector twists a half
    # turn over the loop, giving the Mobius band its single-sided twist.
    theta = seg / SEGMENTS * tau
    radial = (cos(theta), 0.0, sin(theta))
    center = (R * radial[0], 0.0, R * radial[2])
    twist = theta * 0.5 + phase
    # Offset lies in the plane of the radial direction and world up.
    w = (
        WIDTH * cos(twist) * radial[0],
        WIDTH * sin(twist),
        WIDTH * cos(twist) * radial[2],
    )
    left = [center[0] - w[0], center[1] - w[1], center[2] - w[2]]
    right = [center[0] + w[0], center[1] + w[1], center[2] + w[2]]
    return left, right


def setup():
    global p
    size(900, 700)
    window_title(f"Particle surface — {N:,}-vertex Mobius strip")
    mode_3d()

    p = create_particles(capacity=N, attributes=[Attribute.position()])


def draw():
    background(6, 8, 14)

    phase = elapsed_time * 0.25
    positions = []
    for seg in range(SEGMENTS):
        left, right = band(seg, phase)
        positions.append(left)
        positions.append(right)
    p.buffer("position").write(positions)

    t = elapsed_time * 0.15
    r = R * 3.0
    camera_position(cos(t) * r, R * 1.2, sin(t) * r)
    camera_look_at(0.0, 0.0, 0.0)

    particles(p, topology="triangle_strip")  # consecutive vertices -> a surface


run()
