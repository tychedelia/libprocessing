// Per-particle rotation: writes a `rotation` quaternion that rotates the
// configurable `forward` axis to align with the particle's velocity
// direction. Use for boid-like meshes that should "face" their travel
// direction. Default `forward` is +Z, matching `Geometry.box(w, h, d)`'s
// long axis.
//
// Particles with near-zero velocity are skipped (their previous rotation
// is left intact).

struct Params {
    forward: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0) var<storage, read> velocity: array<f32>;
@group(0) @binding(1) var<storage, read_write> rotation: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Shortest-arc quaternion from one unit vector to another. Handles the
// antipodal case (180° rotation) by picking an arbitrary perpendicular
// axis to rotate around.
fn quat_from_to(from: vec3<f32>, to: vec3<f32>) -> vec4<f32> {
    let d = dot(from, to);
    if d > 0.999999 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    if d < -0.999999 {
        var axis = cross(from, vec3<f32>(1.0, 0.0, 0.0));
        if dot(axis, axis) < 0.001 {
            axis = cross(from, vec3<f32>(0.0, 1.0, 0.0));
        }
        return vec4<f32>(normalize(axis), 0.0);
    }
    let axis = cross(from, to);
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

    let from_len2 = dot(params.forward, params.forward);
    if from_len2 < 0.000001 { return; }
    let from = params.forward * inverseSqrt(from_len2);

    let q = quat_from_to(from, dir);

    let ri = i * 4u;
    rotation[ri]      = q.x;
    rotation[ri + 1u] = q.y;
    rotation[ri + 2u] = q.z;
    rotation[ri + 3u] = q.w;
}
