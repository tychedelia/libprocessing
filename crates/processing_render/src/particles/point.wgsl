// Vertex-pulling point-cloud shader for the direct-rasterization path: draws a
// particle `position` storage buffer as `PointList` primitives, one 1px point
// per particle, projected by the camera. No mesh, no vertex buffer — positions
// are pulled by `@builtin(vertex_index)`. See `particles/point_render.rs`.

#import bevy_render::view::View

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<storage, read> positions: array<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let i = vertex_index * 3u;
    let world = vec3<f32>(positions[i], positions[i + 1u], positions[i + 2u]);
    var out: VertexOutput;
    out.clip_position = view.clip_from_world * vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fragment(_in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
