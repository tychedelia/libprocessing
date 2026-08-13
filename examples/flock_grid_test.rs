use processing::prelude::*;

const N: u32 = 300;
const CELL: f32 = 2.5;
const DIMS: [u32; 3] = [4, 4, 4];
const MIN: [f32; 3] = [0.0, 0.0, 0.0];

const COUNT_GRID_SRC: &str = r#"
struct GridParams {
    grid_min: vec3<f32>, cell_size: f32,
    dims_x: u32, dims_y: u32, dims_z: u32, _pad: u32,
}
const RADIUS2: f32 = 6.25;
@group(0) @binding(0) var<storage, read>       position: array<f32>;
@group(0) @binding(1) var<storage, read_write> counts:   array<u32>;
@group(0) @binding(2) var<storage, read>       offsets:  array<u32>;
@group(0) @binding(3) var<storage, read>       sorted:   array<u32>;
@group(0) @binding(4) var<uniform>             gp:       GridParams;

fn load_pos(i: u32) -> vec3<f32> {
    return vec3<f32>(position[i * 3u], position[i * 3u + 1u], position[i * 3u + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&position) / 3u;
    if i >= n { return; }
    let pos = load_pos(i);
    let rel = (pos - gp.grid_min) / gp.cell_size;
    let bx = clamp(i32(floor(rel.x)), 0, i32(gp.dims_x) - 1);
    let by = clamp(i32(floor(rel.y)), 0, i32(gp.dims_y) - 1);
    let bz = clamp(i32(floor(rel.z)), 0, i32(gp.dims_z) - 1);
    var c = 0u;
    for (var dz = -1; dz <= 1; dz++) {
        let cz = bz + dz;
        if cz < 0 || cz >= i32(gp.dims_z) { continue; }
        for (var dy = -1; dy <= 1; dy++) {
            let cy = by + dy;
            if cy < 0 || cy >= i32(gp.dims_y) { continue; }
            for (var dx = -1; dx <= 1; dx++) {
                let cx = bx + dx;
                if cx < 0 || cx >= i32(gp.dims_x) { continue; }
                let cell = u32(cx) + u32(cy) * gp.dims_x + u32(cz) * gp.dims_x * gp.dims_y;
                let start = offsets[cell];
                let end = offsets[cell + 1u];
                for (var s = start; s < end; s++) {
                    let j = sorted[s];
                    if j == i { continue; }
                    let diff = pos - load_pos(j);
                    if dot(diff, diff) < RADIUS2 { c += 1u; }
                }
            }
        }
    }
    counts[i] = c;
}
"#;

const COUNT_BRUTE_SRC: &str = r#"
const RADIUS2: f32 = 6.25;
@group(0) @binding(0) var<storage, read>       position: array<f32>;
@group(0) @binding(1) var<storage, read_write> counts:   array<u32>;

fn load_pos(i: u32) -> vec3<f32> {
    return vec3<f32>(position[i * 3u], position[i * 3u + 1u], position[i * 3u + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&position) / 3u;
    if i >= n { return; }
    let pos = load_pos(i);
    var c = 0u;
    for (var j = 0u; j < n; j++) {
        if j == i { continue; }
        let diff = pos - load_pos(j);
        if dot(diff, diff) < RADIUS2 { c += 1u; }
    }
    counts[i] = c;
}
"#;

fn bytes_to_u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rnd(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    x = (x >> 22) ^ x;
    (x as f32) / (u32::MAX as f32)
}

fn sketch() -> error::Result<bool> {
    init(Config::default())?;
    let surface = surface_create_offscreen(1, 1, 1.0, TextureFormat::Rgba8Unorm)?;
    let _graphics = graphics_create(surface, 1, 1, TextureFormat::Rgba8Unorm)?;

    let mut pos = Vec::with_capacity((N * 3) as usize);
    let mut vel = Vec::with_capacity((N * 3) as usize);
    for i in 0..N {
        for a in 0..3u32 {
            pos.push(0.5 + 9.0 * rnd(i * 3 + a));
            vel.push(0.1 * (rnd(1000 + i * 3 + a) - 0.5));
        }
    }
    let pos_f32 = pos.clone();
    let vel_f32 = vel.clone();

    let f32s_to_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| f.to_le_bytes()).collect() };
    let position = buffer_create_with_data(f32s_to_bytes(&pos))?;
    let velocity = buffer_create_with_data(f32s_to_bytes(&vel))?;
    let counts_grid = buffer_create((N as u64) * 4)?;
    let counts_brute = buffer_create((N as u64) * 4)?;

    let params = GridParams {
        min: MIN,
        cell_size: CELL,
        dims: DIMS,
    };
    let grid = grid_create(params, N)?;
    grid_build(&grid, position)?;

    let mut ok = true;

    let cg = compute_create(shader_create(COUNT_GRID_SRC)?)?;
    compute_set(cg, "position", shader_value::ShaderValue::Buffer(position))?;
    compute_set(cg, "counts", shader_value::ShaderValue::Buffer(counts_grid))?;
    grid_bind(&grid, cg)?;
    compute_dispatch(cg, N.div_ceil(64), 1, 1)?;

    let cb = compute_create(shader_create(COUNT_BRUTE_SRC)?)?;
    compute_set(cb, "position", shader_value::ShaderValue::Buffer(position))?;
    compute_set(cb, "counts", shader_value::ShaderValue::Buffer(counts_brute))?;
    compute_dispatch(cb, N.div_ceil(64), 1, 1)?;

    let g = bytes_to_u32s(&buffer_read(counts_grid)?);
    let b = bytes_to_u32s(&buffer_read(counts_brute)?);
    if g == b {
        let total: u32 = b.iter().sum();
        println!(
            "  PASS neighbour counts match ({} particles, {} total neighbour pairs)",
            N, total
        );
    } else {
        ok = false;
        let i = (0..N as usize).find(|&i| g[i] != b[i]).unwrap();
        println!("  FAIL neighbour count at particle {i}: grid {}, brute {}", g[i], b[i]);
    }

    let flock = particles_kernel_flock()?;
    compute_set(flock, "position", shader_value::ShaderValue::Buffer(position))?;
    compute_set(flock, "velocity", shader_value::ShaderValue::Buffer(velocity))?;
    compute_set(flock, "neighbor_distance", shader_value::ShaderValue::Float(CELL))?;
    grid_bind(&grid, flock)?;
    compute_dispatch(flock, N.div_ceil(64), 1, 1)?;

    let out_vel = bytes_to_f32s(&buffer_read(velocity)?);
    let all_finite = out_vel.iter().all(|v| v.is_finite());
    let changed = out_vel
        .iter()
        .zip(vel_f32.iter())
        .filter(|(a, b)| (**a - **b).abs() > 1e-9)
        .count();
    if all_finite && changed > 0 {
        println!("  PASS flock smoke test ({changed}/{} velocity components changed)", N * 3);
    } else {
        ok = false;
        println!("  FAIL flock smoke test (all_finite={all_finite}, changed={changed})");
    }

    let _ = pos_f32;
    buffer_destroy(position)?;
    buffer_destroy(velocity)?;
    buffer_destroy(counts_grid)?;
    buffer_destroy(counts_brute)?;
    Ok(ok)
}

fn main() {
    let ok = sketch().unwrap();
    if ok {
        println!("flock_grid_test: ALL PASS");
        exit(0).unwrap();
    } else {
        println!("flock_grid_test: FAILURES");
        exit(1).unwrap();
    }
}
