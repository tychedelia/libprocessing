// per-particle rotation quaternion that aligns the configurable `forward`
// axis with the particle's velocity direction. default forward is +Z,
// matching Geometry.box(w, h, d)'s long axis.
//
// near-zero velocity slots are skipped (previous rotation is preserved).

struct Params {
    forward: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> velocity: array<f32>;
@group(0) @binding(1) var<storage, read_write> rotation: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// shortest-arc quaternion from one unit vector to another. antipodal case
// (180°) picks an arbitrary perpendicular axis to rotate around.
fn quat_from_to(src: vec3<f32>, dst: vec3<f32>) -> vec4<f32> {
    let d = dot(src, dst);
    if d > 0.999999 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if d < -0.999999 {
        var axis = cross(src, vec3<f32>(1.0, 0.0, 0.0));
        if dot(axis, axis) < 0.001 {
            axis = cross(src, vec3<f32>(0.0, 1.0, 0.0));
        }
        return vec4<f32>(normalize(axis), 0.0);
    }
    let axis = cross(src, dst);
    return normalize(vec4<f32>(axis, 1.0 + d));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&velocity) / 3u;
    if i >= count { return; }

    let pi = i * 3u;
    let v = vec3<f32>(velocity[pi], velocity[pi + 1u], velocity[pi + 2u]);
    let len2 = dot(v, v);
    if len2 < 0.000001 { return; }
    let dir = v * inverseSqrt(len2);

    let fwd_len2 = dot(params.forward, params.forward);
    if fwd_len2 < 0.000001 { return; }
    let fwd = params.forward * inverseSqrt(fwd_len2);

    let q = quat_from_to(fwd, dir);

    let ri = i * 4u;
    rotation[ri]      = q.x;
    rotation[ri + 1u] = q.y;
    rotation[ri + 2u] = q.z;
    rotation[ri + 3u] = q.w;
}
