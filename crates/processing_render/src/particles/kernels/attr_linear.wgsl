// in-place per-particle scalar linear transform: op = op * scale + offset.
//
// single-slot kernel; destination is read_write. WebGPU forbids binding
// the same buffer as both `read` and `read_write` in one bind group, so
// this is built for the in-place pattern.
//
// op is a reserved namespaced name and won't be auto-bound by
// particles_apply; the caller must explicitly bind via
// `compute.set("op", buffer)`.

struct Params {
    scale: f32,
    offset: f32,
}

@group(0) @binding(0) var<storage, read_write> op: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&op);
    if i >= count { return; }
    op[i] = op[i] * params.scale + params.offset;
}
