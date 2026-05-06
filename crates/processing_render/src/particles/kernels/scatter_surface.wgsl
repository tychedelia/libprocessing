// Initializes particle positions by area-weighted random scatter on a source
// mesh's surface. Sprinkle POP analogue. Dispatched via `particles_emit_gpu`.
//
// Particle attributes required: `position`, `scale`, `age`, `dead`. The
// kernel resets `age = 0` and `dead = 0` so freshly-emitted slots aren't
// skipped by an aging pass on the next frame, and seeds `scale = 1` so a
// follow-up `attr_linear` decay produces the expected fade.
//
// Bindings:
//   source_position: deinterleaved mesh position attribute (3 f32s / vertex).
//     Bound via ShaderValue::MeshAttribute against a deinterleaved Mesh.
//   source_indices: dense u32 index buffer (3 u32s / face). Uploaded by
//     `particles_scatter_create` as a regular storage buffer (mesh slab
//     offsets and u16 indices are flattened CPU-side at setup).
//   cdf: prefix-summed face area, normalized so cdf[face_count-1] == 1.0.
//   position / age / dead: ring-buffer particle attributes.
//   params.face_count: triangle count.
//   emit_range: (base_slot, count, capacity, 0). Matches the other GPU emit
//     kernels' convention; consumed by `particles_emit_gpu`.

struct Params {
    face_count: u32,
    seed: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       source_position: array<f32>;
@group(0) @binding(1) var<storage, read>       source_indices:  array<u32>;
@group(0) @binding(2) var<storage, read>       cdf:             array<f32>;
@group(0) @binding(3) var<storage, read_write> position:        array<f32>;
@group(0) @binding(4) var<storage, read_write> scale:           array<f32>;
@group(0) @binding(5) var<storage, read_write> age:             array<f32>;
@group(0) @binding(6) var<storage, read_write> dead:            array<f32>;
@group(0) @binding(7) var<uniform>             params:          Params;
@group(0) @binding(8) var<uniform>             emit_range:      vec4<f32>;

fn hash(n: u32) -> u32 {
    var x = n;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

fn hash_unit(n: u32) -> f32 {
    return f32(hash(n)) / f32(0xffffffffu);
}

// Branchless-ish binary search for the first index `i` with cdf[i] >= u.
fn cdf_search(u: f32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = params.face_count;
    loop {
        if lo >= hi { break; }
        let mid = (lo + hi) >> 1u;
        if cdf[mid] < u {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return min(lo, params.face_count - 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_i = gid.x;
    if local_i >= u32(emit_range.y) { return; }
    let base = u32(emit_range.x);
    let cap  = u32(emit_range.z);
    let slot = (base + local_i) % cap;
    let seed = params.seed ^ (base + local_i);

    let u01 = hash_unit(seed * 7u + 13u);
    let face = cdf_search(u01);

    let i0 = source_indices[face * 3u + 0u];
    let i1 = source_indices[face * 3u + 1u];
    let i2 = source_indices[face * 3u + 2u];

    let p0 = vec3<f32>(
        source_position[i0 * 3u + 0u],
        source_position[i0 * 3u + 1u],
        source_position[i0 * 3u + 2u],
    );
    let p1 = vec3<f32>(
        source_position[i1 * 3u + 0u],
        source_position[i1 * 3u + 1u],
        source_position[i1 * 3u + 2u],
    );
    let p2 = vec3<f32>(
        source_position[i2 * 3u + 0u],
        source_position[i2 * 3u + 1u],
        source_position[i2 * 3u + 2u],
    );

    // Uniform barycentric on triangle: reflect (u,v) into the lower-left
    // half so the area distribution is uniform.
    var u = hash_unit(seed * 31u + 23u);
    var v = hash_unit(seed * 47u + 29u);
    if u + v > 1.0 { u = 1.0 - u; v = 1.0 - v; }
    let p = (1.0 - u - v) * p0 + u * p1 + v * p2;

    position[slot * 3u + 0u] = p.x;
    position[slot * 3u + 1u] = p.y;
    position[slot * 3u + 2u] = p.z;
    scale[slot * 3u + 0u] = 1.0;
    scale[slot * 3u + 1u] = 1.0;
    scale[slot * 3u + 2u] = 1.0;
    age[slot] = 0.0;
    dead[slot] = 0.0;
}
