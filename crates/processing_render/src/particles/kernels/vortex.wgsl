// vortex around an axis through center. for each particle within radius
// (perpendicular to the axis), adds a tangential impulse in direction
// cross(axis, radial). right-hand rule: positive strength rotates
// counter-clockwise looking along the negative axis direction.
//
// falloff_mode matches Attract: 0 = constant, 1 = linear, 2 = smoothstep,
// 3 = inverse-distance.

struct Params {
    center: vec3<f32>,
    _pad0: f32,
    axis: vec3<f32>,
    _pad1: f32,
    strength: f32,
    radius: f32,
    falloff_mode: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> position: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&position) / 3u;
    if i >= count { return; }

    let axis_len2 = dot(params.axis, params.axis);
    if axis_len2 < 0.000001 { return; }
    let axis = params.axis * inverseSqrt(axis_len2);

    let pi = i * 3u;
    let pos = vec3<f32>(position[pi], position[pi + 1u], position[pi + 2u]);
    let offset = pos - params.center;

    // perpendicular component determines the radius from the axis line.
    let parallel = dot(offset, axis) * axis;
    let radial = offset - parallel;
    let r2 = dot(radial, radial);
    let max_r2 = params.radius * params.radius;
    if r2 > max_r2 || r2 < 0.000001 { return; }

    let r = sqrt(r2);
    let tangent = cross(axis, radial / r);

    var fall: f32 = 1.0;
    let n = 1.0 - r / params.radius;
    if params.falloff_mode == 1u {
        fall = n;
    } else if params.falloff_mode == 2u {
        fall = n * n * (3.0 - 2.0 * n);
    } else if params.falloff_mode == 3u {
        fall = 1.0 / (r + 1.0);
    }

    let kick = tangent * (params.strength * fall);
    velocity[pi]      = velocity[pi]      + kick.x;
    velocity[pi + 1u] = velocity[pi + 1u] + kick.y;
    velocity[pi + 2u] = velocity[pi + 2u] + kick.z;
}
