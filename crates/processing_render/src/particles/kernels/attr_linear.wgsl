// In-place per-particle scalar linear transform: op = op * scale + offset.
//
// Single-slot kernel: the destination attribute is bound to `op` (a
// read_write storage). A WebGPU bind group can't hold the same buffer
// as both `read` and `read_write` storage, so this kernel is built for
// the in-place pattern (the dominant use case — decay, scale, fill).
//
// With scale=1, offset=0 it's a no-op identity. With scale=0.965,
// offset=0 it's a 3.5% per-dispatch decay. With scale=0, offset=k it
// fills every slot with the constant k.
//
// `op` is a reserved namespaced name; it won't be auto-bound by
// `particles_apply`, so the user must explicitly bind it via
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
