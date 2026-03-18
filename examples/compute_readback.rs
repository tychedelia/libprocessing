use processing::prelude::*;

fn main() {
    match run() {
        Ok(_) => {
            eprintln!("Compute readback test passed!");
            exit(0).unwrap();
        }
        Err(e) => {
            eprintln!("Compute readback error: {:?}", e);
            exit(1).unwrap();
        }
    }
}

fn run() -> error::Result<()> {
    init(Config::default())?;

    let surface = surface_create_offscreen(1, 1, 1.0, TextureFormat::Rgba8Unorm)?;
    let graphics = graphics_create(surface, 1, 1, TextureFormat::Rgba8Unorm)?;

    let buf = buffer_create(16)?;

    graphics_begin_draw(graphics)?;
    graphics_end_draw(graphics)?;

    graphics_begin_draw(graphics)?;
    graphics_end_draw(graphics)?;

    let shader_src = r#"
@group(0) @binding(0)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(1)
fn main() {
    output[0] = 1u;
    output[1] = 2u;
    output[2] = 3u;
    output[3] = 4u;
}
"#;
    let shader = shader_create(shader_src)?;
    let compute = compute_create(shader)?;
    compute_set(compute, "output", shader_value::ShaderValue::Buffer(buf))?;

    compute_dispatch(compute, 1, 1, 1)?;

    let data = buffer_read(buf)?;
    let values: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    assert_eq!(values, vec![1, 2, 3, 4], "Compute readback mismatch!");

    compute_destroy(compute)?;
    shader_destroy(shader)?;
    buffer_destroy(buf)?;

    Ok(())
}
