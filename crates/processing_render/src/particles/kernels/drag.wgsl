// velocity damping. each dispatch, velocity *= (1 - coefficient).
// velocity_cap > 0 also clamps the resulting speed.

struct Params {
    coefficient: f32,
    velocity_cap: f32,
}

@group(0) @binding(0) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&velocity) / 3u;
    if i >= count { return; }

    let pi = i * 3u;
    let v = vec3<f32>(velocity[pi], velocity[pi + 1u], velocity[pi + 2u]);
    var new_v = v * (1.0 - params.coefficient);
    if params.velocity_cap > 0.0 {
        let speed2 = dot(new_v, new_v);
        let cap2 = params.velocity_cap * params.velocity_cap;
        if speed2 > cap2 {
            new_v = new_v * (params.velocity_cap * inverseSqrt(speed2));
        }
    }
    velocity[pi]      = new_v.x;
    velocity[pi + 1u] = new_v.y;
    velocity[pi + 2u] = new_v.z;
}
