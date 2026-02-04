mod glfw;

use glfw::GlfwContext;
use processing::prelude::*;
use processing_render::render::command::DrawCommand;

fn main() {
    match sketch() {
        Ok(_) => {
            eprintln!("Sketch completed successfully");
            exit(0).unwrap();
        }
        Err(e) => {
            eprintln!("Sketch error: {:?}", e);
            exit(1).unwrap();
        }
    };
}

fn sketch() -> error::Result<()> {
    let width = 600;
    let height = 600;
    let mut glfw_ctx = GlfwContext::new(width, height)?;
    init(Config::default())?;

    let surface = glfw_ctx.create_surface(width, height, 1.0)?;
    let graphics = graphics_create(surface, width, height)?;
    let cube = geometry_box(60.0, 60.0, 60.0)?;

    graphics_mode_3d(graphics)?;
    graphics_camera_position(graphics, 250.0, 250.0, 500.0)?;
    graphics_camera_look_at(graphics, 0.0, 0.0, 0.0)?;

    // Enable bloom so emissive cubes glow, and TonyMcMapface tonemapping
    graphics_bloom(graphics, 0.3)?;
    graphics_tonemapping(graphics, 6)?;

    // Create a 5x5 grid of materials varying roughness (x) and metallic (y).
    // We use emissive color so the cubes are visible without scene lights.
    let grid = 5;
    let mut materials = Vec::new();
    for row in 0..grid {
        for col in 0..grid {
            let mat = material_create_pbr()?;

            let roughness = col as f32 / (grid - 1) as f32;
            let metallic = row as f32 / (grid - 1) as f32;

            material_set(mat, "roughness", material::MaterialValue::Float(roughness))?;
            material_set(mat, "metallic", material::MaterialValue::Float(metallic))?;

            // Emissive so the cubes are self-lit
            let intensity = 2.0;
            let r = 0.2 + 0.8 * (1.0 - roughness);
            let g = 0.2 + 0.6 * metallic;
            let b = 0.6;
            material_set(
                mat,
                "emissive",
                material::MaterialValue::Float4([r * intensity, g * intensity, b * intensity, 1.0]),
            )?;

            materials.push(mat);
        }
    }

    let mut angle: f32 = 0.0;
    let spacing = 90.0;
    let offset = (grid - 1) as f32 * spacing / 2.0;

    while glfw_ctx.poll_events() {
        graphics_begin_draw(graphics)?;

        graphics_record_command(
            graphics,
            DrawCommand::BackgroundColor(bevy::color::Color::srgb(0.05, 0.05, 0.08)),
        )?;

        for row in 0..grid {
            for col in 0..grid {
                let mat = materials[row * grid + col];

                graphics_record_command(graphics, DrawCommand::PushMatrix)?;
                graphics_record_command(
                    graphics,
                    DrawCommand::Translate {
                        x: col as f32 * spacing - offset,
                        y: row as f32 * spacing - offset,
                    },
                )?;
                graphics_record_command(graphics, DrawCommand::Rotate { angle })?;
                graphics_record_command(graphics, DrawCommand::Material(mat))?;
                graphics_record_command(graphics, DrawCommand::Geometry(cube))?;
                graphics_record_command(graphics, DrawCommand::PopMatrix)?;
            }
        }

        graphics_end_draw(graphics)?;
        angle += 0.01;
    }

    for mat in materials {
        material_destroy(mat)?;
    }

    Ok(())
}
