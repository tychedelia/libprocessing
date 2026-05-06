// Sprinkle "Volume" mode analogue: rejection-sampled particles filling the
// interior of a closed mesh. Uses the Duck.glb so the volume reads as a
// recognizable shape rather than an abstract blob.

use processing_glfw::GlfwContext;
use std::time::Instant;

use bevy::math::Vec3;
use processing::prelude::*;
use processing_render::geometry::AttributeFormat;
use processing_render::render::command::DrawCommand;

const AGE_SHADER: &str = r#"
struct Params { dt: f32, ttl: f32, _pad0: f32, _pad1: f32 }

@group(0) @binding(0) var<storage, read_write> scale: array<f32>;
@group(0) @binding(1) var<storage, read_write> age:   array<f32>;
@group(0) @binding(2) var<storage, read_write> dead:  array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = arrayLength(&age);
    if i >= count { return; }
    if dead[i] != 0.0 { return; }

    age[i] = age[i] + params.dt;
    let t = age[i] / params.ttl;
    let rise = clamp(t / 0.05, 0.0, 1.0);
    let fall = clamp((1.0 - t) / 0.6, 0.0, 1.0);
    let s = rise * fall;
    scale[i * 3u + 0u] = s;
    scale[i * 3u + 1u] = s;
    scale[i * 3u + 2u] = s;

    if age[i] > params.ttl { dead[i] = 1.0; }
}
"#;

fn main() {
    sketch().unwrap();
    exit(0).unwrap();
}

fn sketch() -> error::Result<()> {
    let mut glfw_ctx = GlfwContext::new(600, 400)?;
    init(Config::default())?;

    let surface = glfw_ctx.create_surface(900, 700)?;
    let graphics = graphics_create(surface, 900, 700, TextureFormat::Rgba16Float)?;

    graphics_mode_3d(graphics)?;
    // Duck.glb is authored in cm — bbox is roughly 150 wide × 200 tall. Frame
    // it from a distance that fits, then enable the interactive orbit camera
    // so you can drag-rotate around the volume.
    transform_set_position(graphics, Vec3::new(0.0, 100.0, 400.0))?;
    transform_look_at(graphics, Vec3::new(0.0, 80.0, 0.0))?;
    graphics_orbit_camera(graphics)?;

    // Source mesh: the Duck. Its volume — body, head, beak — is what particles
    // fill via AABB rejection sampling.
    let gltf = gltf_load(graphics, "gltf/Duck.glb")?;
    let duck = gltf_geometry(gltf, "LOD3spShape")?;
    let scatter = particles_scatter_volume_create(duck)?;

    let particle = geometry_sphere(0.15, 4, 3)?;

    let capacity: u32 = 30_000;
    let position_attr = geometry_attribute_position();
    let scale_attr = geometry_attribute_scale();
    let dead_attr = geometry_attribute_dead();
    let age_attr = geometry_attribute_create("age", AttributeFormat::Float)?;

    let p = particles_create(
        capacity,
        vec![position_attr, scale_attr, dead_attr, age_attr],
    )?;

    let dead_buf = particles_buffer(p, dead_attr)?
        .ok_or(error::ProcessingError::ParticlesNotFound)?;
    let init_dead: Vec<u8> = (0..capacity).flat_map(|_| 1.0_f32.to_le_bytes()).collect();
    buffer_write(dead_buf, init_dead)?;

    let age_shader = shader_create(AGE_SHADER)?;
    let aging = compute_create(age_shader)?;

    let mat = material_create_unlit()?;
    material_set_albedo_color(mat, [1.0, 1.0, 1.0, 1.0])?;

    // Volume scatter is O(faces × attempts) per particle, so we keep the
    // burst modest. Capacity × ttl⁻¹ should match — here ~30K alive.
    let burst: u32 = 250;
    let dt: f32 = 1.0 / 60.0;
    let ttl: f32 = 5.0;
    let start = Instant::now();

    while glfw_ctx.poll_events() {
        graphics_begin_draw(graphics)?;
        graphics_record_command(
            graphics,
            DrawCommand::BackgroundColor(bevy::color::Color::srgb(0.03, 0.03, 0.05)),
        )?;
        graphics_record_command(graphics, DrawCommand::Material(mat))?;
        graphics_record_command(
            graphics,
            DrawCommand::Particles {
                particles: p,
                geometry: particle,
            },
        )?;
        graphics_end_draw(graphics)?;

        let t = start.elapsed().as_secs_f32();

        compute_set(
            scatter,
            "seed",
            shader_value::ShaderValue::UInt((t * 1000.0) as u32 ^ 0xc0ffeeu32),
        )?;
        particles_emit_gpu(p, burst, scatter)?;

        compute_set(aging, "dt", shader_value::ShaderValue::Float(dt))?;
        compute_set(aging, "ttl", shader_value::ShaderValue::Float(ttl))?;
        particles_apply(p, aging)?;
    }

    Ok(())
}
