// Per-particle 1D ramp lookup: out = sample(ramp, t) where
// t = clamp(in_attr * scale + offset, 0, 1).
//
// Generic-slot kernel. The user binds:
//   - `op_in`: scalar (f32) input attribute used as the lookup coord
//   - `ramp`: a texture (typically 1D or N×1) used as the ramp
//   - `op_out_*`: one or more scalar output attributes for the sampled
//     channels (we expose r/g/b/a as separate slots so the caller can
//     write into separate Float attributes — or omit channels they
//     don't need by binding the same buffer as a scratch).
//
// To skip channels you don't care about, bind the same buffer to
// multiple `op_out_*` slots — bevy_naga_reflect only complains if a
// declared slot is unbound, not if a buffer is over-bound. (Or, for a
// cleaner setup, define an unused scratch attribute on the particle
// system and bind it.)
//
// `ramp` is sampled as a 2D texture at v=0.5 so it works with both 2D
// 1×N gradient textures (the common case in Processing) and proper 1D
// textures (read as 2D with height 1).

struct Params {
    scale: f32,
    offset: f32,
}

@group(0) @binding(0) var<storage, read> op_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> op_out_r: array<f32>;
@group(0) @binding(2) var<storage, read_write> op_out_g: array<f32>;
@group(0) @binding(3) var<storage, read_write> op_out_b: array<f32>;
@group(0) @binding(4) var<storage, read_write> op_out_a: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var ramp: texture_2d<f32>;
@group(0) @binding(7) var ramp_sampler: sampler;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&op_in);
    if i >= count { return; }

    let t = clamp(op_in[i] * params.scale + params.offset, 0.0, 1.0);
    let c = textureSampleLevel(ramp, ramp_sampler, vec2<f32>(t, 0.5), 0.0);
    op_out_r[i] = c.r;
    op_out_g[i] = c.g;
    op_out_b[i] = c.b;
    op_out_a[i] = c.a;
}
