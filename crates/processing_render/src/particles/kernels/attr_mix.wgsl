// In-place per-particle scalar lerp:
//   op_a = mix(op_a, op_b, clamp(op_t * t_scale + t_offset, 0, 1)).
//
// `op_a` is the destination (read_write); `op_b` and `op_t` are
// read-only. WebGPU disallows aliasing a read_write buffer with
// another binding in the same bind group, so `op_a`, `op_b`, and
// `op_t` must point at distinct buffers. Set `t_clamp = 0u` to skip
// the [0, 1] clamp on `t` (useful for additive blends).
//
// Common patterns:
//   - Conditional write: `op_t` is 0 outside a region, 1 inside (e.g.
//     output of `kernelField` with hard falloff); `op_b` is the new
//     value. Result: only inside-region particles get `op_b`.
//   - Soft fade: `op_t` is a continuous falloff weight; `op_a` /
//     `op_b` are start/end states.
//
// `op_a`, `op_b`, `op_t` are reserved namespaced names; they won't be
// auto-bound by `particles_apply`.

struct Params {
    t_scale: f32,
    t_offset: f32,
    t_clamp: u32,
}

@group(0) @binding(0) var<storage, read_write> op_a: array<f32>;
@group(0) @binding(1) var<storage, read> op_b: array<f32>;
@group(0) @binding(2) var<storage, read> op_t: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&op_a);
    if i >= count { return; }

    var t = op_t[i] * params.t_scale + params.t_offset;
    if params.t_clamp != 0u {
        t = clamp(t, 0.0, 1.0);
    }
    op_a[i] = mix(op_a[i], op_b[i], t);
}
