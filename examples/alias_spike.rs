//! Spike for the attribute-algebra `dst == src` question (task #4).
//!
//! The new verbs (map/combine/mix) take an explicit `dst` that may alias an
//! input. This probes what the wgpu/Metal backend actually does when the *same*
//! buffer is bound to both a `read` binding and a `read_write` binding in one
//! dispatch — vs. the single `read_write` binding used for genuine in-place.
//!
//! Run: `cargo run --example alias_spike`. Interpret the printed results:
//!   - distinct buffers: sanity baseline, must be [2,4,6,8].
//!   - aliased read+read_write: if this comes back [2,4,6,8] with no validation
//!     error, dual-binding aliasing is allowed and one kernel covers both cases.
//!     If it errors / returns unchanged, aliasing is rejected → in-place must be
//!     a single read_write binding.
//!   - single read_write in-place: the fallback path; must be [2,4,6,8].

use processing::prelude::*;

const TWO_BINDING_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&dst) { return; }
    dst[i] = src[i] * 2.0;
}
"#;

const ONE_BINDING_SRC: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&data) { return; }
    data[i] = data[i] * 2.0;
}
"#;

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn run() -> error::Result<()> {
    init(Config::default())?;
    let surface = surface_create_offscreen(1, 1, 1.0, TextureFormat::Rgba8Unorm)?;
    let _graphics = graphics_create(surface, 1, 1, TextureFormat::Rgba8Unorm)?;

    let base = [1.0f32, 2.0, 3.0, 4.0];

    // Case A: distinct buffers (baseline).
    let two = compute_create(shader_create(TWO_BINDING_SRC)?)?;
    let a_src = buffer_create_with_data(to_bytes(&base))?;
    let a_dst = buffer_create_with_data(to_bytes(&[0.0; 4]))?;
    compute_set(two, "src", shader_value::ShaderValue::Buffer(a_src))?;
    compute_set(two, "dst", shader_value::ShaderValue::Buffer(a_dst))?;
    compute_dispatch(two, 1, 1, 1)?;
    println!("A distinct read+read_write  -> {:?}", f32s(&buffer_read(a_dst)?));

    // Case C: single read_write binding, genuine in-place. (Run before the
    // fatal aliasing case so its result is observable.)
    let one = compute_create(shader_create(ONE_BINDING_SRC)?)?;
    let c = buffer_create_with_data(to_bytes(&base))?;
    compute_set(one, "data", shader_value::ShaderValue::Buffer(c))?;
    compute_dispatch(one, 1, 1, 1)?;
    println!("C single read_write in-place-> {:?} (want [2,4,6,8])", f32s(&buffer_read(c)?));

    // Case B: SAME buffer bound to both the read and the read_write binding.
    // Expect a *fatal* wgpu validation error — this is the whole finding.
    let b = buffer_create_with_data(to_bytes(&base))?;
    compute_set(two, "src", shader_value::ShaderValue::Buffer(b))?;
    compute_set(two, "dst", shader_value::ShaderValue::Buffer(b))?;
    println!("B aliased read+read_write   -> dispatching (expect fatal Validation Error)...");
    compute_dispatch(two, 1, 1, 1)?;
    println!("B aliased read+read_write   -> {:?} (want [2,4,6,8])", f32s(&buffer_read(b)?));

    Ok(())
}

fn main() {
    run().unwrap();
    exit(0).unwrap();
}
