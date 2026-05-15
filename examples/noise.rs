//! Visualizes the four noise kinds as grayscale tile grids.
//!
//! Keys:
//!   1 = Perlin
//!   2 = Simplex
//!   3 = Value
//!   4 = Worley (nearest)
//!   5 = Worley (second nearest)
//!   6 = Worley (difference)
//!   M = cycle Worley distance metric (Euclidean / Manhattan / Chebyshev)
//!   [ / ] = decrease / increase frequency
//!   - / = = decrease / increase octaves
//!   , / . = decrease / increase persistence
//!   R = randomize seed
//!   ESC = quit

use processing::prelude::*;
use processing_glfw::GlfwContext;
use processing_render::render::command::DrawCommand;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const CELLS: u32 = 128;

fn main() {
    match sketch() {
        Ok(_) => exit(0).unwrap(),
        Err(e) => {
            eprintln!("Sketch error: {e:?}");
            exit(1).unwrap();
        }
    }
}

fn sketch() -> error::Result<()> {
    let mut glfw_ctx = GlfwContext::new(WIDTH, HEIGHT)?;
    init(Config::default())?;

    let surface = glfw_ctx.create_surface(WIDTH, HEIGHT)?;
    let graphics = graphics_create(surface, WIDTH, HEIGHT, TextureFormat::Rgba16Float)?;

    let n = noise_create()?;
    noise_frequency(n, 4.0)?;

    let cell = WIDTH as f32 / CELLS as f32;
    let inv = 1.0 / CELLS as f32;

    let mut frequency = 4.0_f32;
    let mut octaves = 4_u32;
    let mut persistence = 0.5_f32;
    let mut distance_idx = 0_u32;
    let mut seed = 0_u32;
    let mut t = 0.0_f32;

    while glfw_ctx.poll_events() {
        if input_key_is_down(KeyCode::Escape)? {
            break;
        }

        if input_key_just_pressed(KeyCode::Digit1)? {
            noise_mode(n, NoiseKind::Perlin)?;
        }
        if input_key_just_pressed(KeyCode::Digit2)? {
            noise_mode(n, NoiseKind::Simplex)?;
        }
        if input_key_just_pressed(KeyCode::Digit3)? {
            noise_mode(n, NoiseKind::Value)?;
        }
        if input_key_just_pressed(KeyCode::Digit4)? {
            noise_mode(n, NoiseKind::Worley)?;
            noise_worley(n, WorleyMode::Nearest)?;
        }
        if input_key_just_pressed(KeyCode::Digit5)? {
            noise_mode(n, NoiseKind::Worley)?;
            noise_worley(n, WorleyMode::SecondNearest)?;
        }
        if input_key_just_pressed(KeyCode::Digit6)? {
            noise_mode(n, NoiseKind::Worley)?;
            noise_worley(n, WorleyMode::Difference)?;
        }
        if input_key_just_pressed(KeyCode::KeyM)? {
            distance_idx = (distance_idx + 1) % 3;
            let dist = match distance_idx {
                0 => NoiseDistance::Euclidean,
                1 => NoiseDistance::Manhattan,
                _ => NoiseDistance::Chebyshev,
            };
            noise_distance(n, dist)?;
        }
        if input_key_just_pressed(KeyCode::BracketLeft)? {
            frequency = (frequency * 0.5).max(0.25);
            noise_frequency(n, frequency)?;
        }
        if input_key_just_pressed(KeyCode::BracketRight)? {
            frequency = (frequency * 2.0).min(64.0);
            noise_frequency(n, frequency)?;
        }
        if input_key_just_pressed(KeyCode::Minus)? {
            octaves = octaves.saturating_sub(1).max(1);
            noise_detail(n, octaves, persistence)?;
        }
        if input_key_just_pressed(KeyCode::Equal)? {
            octaves = (octaves + 1).min(10);
            noise_detail(n, octaves, persistence)?;
        }
        if input_key_just_pressed(KeyCode::Comma)? {
            persistence = (persistence - 0.1).max(0.0);
            noise_detail(n, octaves, persistence)?;
        }
        if input_key_just_pressed(KeyCode::Period)? {
            persistence = (persistence + 0.1).min(1.0);
            noise_detail(n, octaves, persistence)?;
        }
        if input_key_just_pressed(KeyCode::KeyR)? {
            seed = seed.wrapping_add(1);
            noise_seed(n, seed)?;
        }

        graphics_begin_draw(graphics)?;
        graphics_record_command(
            graphics,
            DrawCommand::BackgroundColor(bevy::color::Color::srgb(0.0, 0.0, 0.0)),
        )?;
        graphics_record_command(graphics, DrawCommand::NoStroke)?;

        for j in 0..CELLS {
            for i in 0..CELLS {
                let x = i as f32 * inv;
                let y = j as f32 * inv;
                let raw = noise_sample_3d(n, x, y, t)?;
                // Perlin/Simplex output [-1, 1]; Value/Worley output [0, 1]. Map either to [0, 1].
                let v = (raw * 0.5 + 0.5).clamp(0.0, 1.0);
                graphics_record_command(
                    graphics,
                    DrawCommand::Fill(bevy::color::Color::srgb(v, v, v)),
                )?;
                graphics_record_command(
                    graphics,
                    DrawCommand::Rect {
                        x: i as f32 * cell,
                        y: j as f32 * cell,
                        w: cell,
                        h: cell,
                        radii: [0.0; 4],
                    },
                )?;
            }
        }

        graphics_end_draw(graphics)?;
        t += 0.01;
    }

    noise_destroy(n)?;
    Ok(())
}
