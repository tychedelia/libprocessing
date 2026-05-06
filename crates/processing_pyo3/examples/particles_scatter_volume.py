# Sprinkle "Volume" mode analogue: rejection-sampled particles filling the
# interior of the Duck.glb. AABB-uniform random points get parity-tested
# against the mesh on the GPU; odd hits = inside, accept.
#
# Decay uses the built-in `attr_linear` kernel (op = op * scale + offset)
# applied to the per-particle scale buffer — no custom WGSL needed. The
# scatter kernel seeds `scale = 1` on emit; attr_linear shrinks it each
# frame, and `life` zero-crossing handles culling automatically.

from mewnala import *

CAPACITY = 30_000
BURST = 250

p = None
particle = None
mat = None
scatter = None
decay = None


def setup():
    global p, particle, mat, scatter, decay

    size(900, 700)
    mode_3d()

    # Duck.glb is authored in cm — bbox is roughly 150 wide × 200 tall.
    camera_position(0.0, 100.0, 400.0)
    camera_look_at(0.0, 80.0, 0.0)
    orbit_camera()

    gltf = load_gltf("gltf/Duck.glb")
    duck = gltf.geometry("LOD3spShape")
    scatter = Particles.scatter_volume(duck)

    particle = Geometry.sphere(0.15, 4, 3)

    # `age` and `life` are required by the scatter kernel even though aging
    # itself is handled implicitly by attr_linear shrinking scale.
    age_attr = Attribute("age", AttributeFormat.Float)
    p = Particles(
        capacity=CAPACITY,
        attributes=[
            Attribute.position(),
            Attribute.scale(),
            Attribute.life(),
            age_attr,
        ],
    )
    # Zero-fill of `life` parks unemitted slots automatically (life=0 = culled).

    mat = Material.unlit(albedo=[1.0, 1.0, 1.0, 1.0])

    # Geometric decay on the scale buffer: scale ← scale × 0.985 / frame.
    # `op` is the reserved single-slot binding the kernel writes through.
    decay = Particles.attr_linear()
    decay.set(op=p.buffer(Attribute.scale()), scale=0.985, offset=0.0)


def draw():
    background(8, 8, 13)
    use_material(mat)
    particles(p, particle)

    seed = (int(elapsed_time * 1000.0) ^ 0xC0FFEE) & 0xFFFFFFFF
    scatter.set(seed=seed)
    p.emit_gpu(BURST, scatter)
    p.apply(decay)


run()
