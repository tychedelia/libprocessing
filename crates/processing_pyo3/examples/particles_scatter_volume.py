# Sprinkle "Volume" mode analogue: rejection-sampled particles filling the
# interior of the Duck.glb. AABB-uniform random points get parity-tested
# against the mesh on the GPU; odd hits = inside, accept.

from mewnala import *

CAPACITY = 30_000
BURST = 250
DT = 1.0 / 60.0
TTL = 5.0

AGE_SHADER = """
struct Params { dt: f32, ttl: f32, _pad0: f32, _pad1: f32 }

@group(0) @binding(0) var<storage, read_write> scale: array<f32>;
@group(0) @binding(1) var<storage, read_write> age:   array<f32>;
@group(0) @binding(2) var<storage, read_write> life:  array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&age);
    if i >= count { return; }
    if life[i] <= 0.0 { return; }

    age[i] = age[i] + params.dt;
    let t = age[i] / params.ttl;
    let rise = clamp(t / 0.05, 0.0, 1.0);
    let fall = clamp((1.0 - t) / 0.6, 0.0, 1.0);
    let s = rise * fall;
    scale[i * 3u + 0u] = s;
    scale[i * 3u + 1u] = s;
    scale[i * 3u + 2u] = s;

    if age[i] > params.ttl { life[i] = 0.0; }
}
"""

p = None
particle = None
mat = None
scatter = None
aging = None


def setup():
    global p, particle, mat, scatter, aging

    size(900, 700)
    mode_3d()

    # Duck.glb is authored in cm — bbox is roughly 150 wide × 200 tall.
    camera_position(0.0, 100.0, 400.0)
    camera_look_at(0.0, 80.0, 0.0)
    orbit_camera()

    gltf = load_gltf("gltf/Duck.glb")
    duck = gltf.geometry("LOD3spShape")
    scatter = kernel_scatter_volume(duck)

    particle = Geometry.sphere(0.15, 4, 3)

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

    aging = Compute(Shader(AGE_SHADER))


def draw():
    background(8, 8, 13)
    use_material(mat)
    particles(p, particle)

    seed = (int(elapsed_time * 1000.0) ^ 0xC0FFEE) & 0xFFFFFFFF
    scatter.set(seed=seed)
    p.emit_gpu(BURST, scatter)

    p.apply(aging, dt=DT, ttl=TTL)


run()
