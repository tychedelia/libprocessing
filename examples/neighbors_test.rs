//! Neighbor lists and reductions against brute force.

use std::collections::BTreeSet;

use bevy::prelude::Entity;
use processing::prelude::*;
use processing_render::{
    FALLOFF_CONST, FALLOFF_INVERSE_SQUARE, GridParams, grid_create, particles_find_neighbors,
    particles_neighbor_lists, particles_neighbor_reduce,
};

const N: u32 = 400;
const CELL: f32 = 2.5;

// sums each particle's neighbor indices; `apply` binds the lists by name
const SUM_SRC: &str = r#"
@group(0) @binding(0) var<storage, read>       neighbors:      array<u32>;
@group(0) @binding(1) var<storage, read>       neighbor_count: array<u32>;
@group(0) @binding(2) var<storage, read_write> out:            array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = arrayLength(&neighbor_count);
    if i >= n { return; }
    let first = i * (arrayLength(&neighbors) / n);
    var sum = 0u;
    for (var k = 0u; k < neighbor_count[i]; k++) {
        sum += neighbors[first + k];
    }
    out[i] = sum;
}
"#;

fn rnd(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    x = (x >> 22) ^ x;
    (x as f32) / (u32::MAX as f32)
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn dist2(pos: &[f32], i: usize, j: usize) -> f32 {
    (0..3)
        .map(|a| (pos[j * 3 + a] - pos[i * 3 + a]).powi(2))
        .sum()
}

fn lists(particles: Entity) -> error::Result<Vec<Vec<u32>>> {
    let lists = particles_neighbor_lists(particles)?.expect("find_neighbors ran");
    let index = u32s(&buffer_read(lists.neighbors)?);
    let count = u32s(&buffer_read(lists.count)?);
    Ok((0..N as usize)
        .map(|i| index[i * lists.max as usize..][..count[i] as usize].to_vec())
        .collect())
}

fn sketch() -> error::Result<bool> {
    init(Config::default())?;
    let surface = surface_create_offscreen(1, 1, 1.0, TextureFormat::Rgba8Unorm)?;
    let _graphics = graphics_create(surface, 1, 1, TextureFormat::Rgba8Unorm)?;

    let pos: Vec<f32> = (0..N * 3).map(|k| 0.5 + 9.0 * rnd(k)).collect();
    let vel: Vec<f32> = (0..N * 3).map(|k| rnd(5000 + k) - 0.5).collect();
    let (p_attr, v_attr) = (geometry_attribute_position(), geometry_attribute_velocity());
    let particles = particles_create(N, vec![p_attr, v_attr])?;
    let position = particles_buffer(particles, p_attr)?.unwrap();
    let velocity = particles_buffer(particles, v_attr)?.unwrap();
    buffer_write(position, bytes(&pos))?;
    buffer_write(velocity, bytes(&vel))?;
    let grid = grid_create(
        GridParams {
            min: [0.0; 3],
            cell_size: CELL,
            dims: [4, 4, 4],
        },
        N,
    )?;
    let mut ok = true;
    let mut check = |pass: bool, what: String| {
        println!("  {} {what}", if pass { "PASS" } else { "FAIL" });
        ok &= pass;
    };

    // radius 3.5 reaches past the neighboring cells
    for radius in [CELL, 3.5] {
        particles_find_neighbors(particles, grid, radius, N)?;
        let got = lists(particles)?;
        let exact = (0..N as usize).all(|i| {
            let want: BTreeSet<u32> = (0..N as usize)
                .filter(|&j| j != i && dist2(&pos, i, j) <= radius * radius)
                .map(|j| j as u32)
                .collect();
            got[i].iter().copied().collect::<BTreeSet<_>>() == want && got[i].len() == want.len()
        });
        let pairs: usize = got.iter().map(Vec::len).sum();
        check(
            exact,
            format!("lists match brute force at radius {radius} ({pairs} pairs)"),
        );
    }

    particles_find_neighbors(particles, grid, CELL, 8)?;
    let got = lists(particles)?;
    let valid = got.iter().enumerate().all(|(i, list)| {
        let unique: BTreeSet<_> = list.iter().collect();
        list.len() <= 8
            && unique.len() == list.len()
            && list
                .iter()
                .all(|&j| j as usize != i && dist2(&pos, i, j as usize) <= CELL * CELL)
    });
    let full = got.iter().filter(|l| l.len() == 8).count();
    check(valid, format!("max 8: valid lists, {full}/{N} full"));

    // reductions over the exact lists
    particles_find_neighbors(particles, grid, CELL, N)?;
    let all = lists(particles)?;
    let out = buffer_create((N * 3 * 4) as u64)?;
    let r = 2.0f32;
    let cases = [
        ("sum velocity", velocity, 0, FALLOFF_CONST, false),
        ("mean velocity", velocity, 1, FALLOFF_CONST, false),
        (
            "relative position, inverse square",
            position,
            0,
            FALLOFF_INVERSE_SQUARE,
            true,
        ),
    ];
    for (what, source, op, falloff, relative) in cases {
        particles_neighbor_reduce(particles, source, out, op, r, falloff, 3, relative)?;
        let got = f32s(&buffer_read(out)?);
        let src = if source == velocity { &vel } else { &pos };
        let mut worst = 0.0f32;
        for i in 0..N as usize {
            let (mut value, mut wsum) = ([0.0f32; 3], 0.0f32);
            for &j in &all[i] {
                let d2 = dist2(&pos, i, j as usize);
                if d2 > r * r {
                    continue;
                }
                let w = if falloff == FALLOFF_CONST {
                    1.0
                } else {
                    1.0 / d2.max(1e-6)
                };
                wsum += w;
                for a in 0..3 {
                    let own = if relative { src[i * 3 + a] } else { 0.0 };
                    value[a] += w * (src[j as usize * 3 + a] - own);
                }
            }
            for a in 0..3 {
                let want = if op == 1 && wsum > 0.0 {
                    value[a] / wsum
                } else {
                    value[a]
                };
                worst = worst.max((got[i * 3 + a] - want).abs() / want.abs().max(1.0));
            }
        }
        check(
            worst < 1e-4,
            format!("{what} matches the CPU (max rel diff {worst:e})"),
        );
    }

    let summer = compute_create(shader_create(SUM_SRC)?)?;
    let sums = buffer_create((N * 4) as u64)?;
    compute_set(summer, "out", shader_value::ShaderValue::Buffer(sums))?;
    particles_apply(particles, summer)?;
    let got = u32s(&buffer_read(sums)?);
    let same = (0..N as usize).all(|i| got[i] == all[i].iter().sum::<u32>());
    check(same, "a custom kernel reads the lists by name".to_string());

    // `radius` names a field in two uniform structs
    let clash = compute_create(shader_create(
        r#"
struct Params {
    radius: f32,
}

struct Other {
    radius: f32,
}

@group(0) @binding(0) var<uniform>             params: Params;
@group(0) @binding(1) var<uniform>             other:  Other;
@group(0) @binding(2) var<storage, read_write> out:    array<f32>;

@compute @workgroup_size(1)
fn main() {
    out[0] = params.radius + other.radius * 0.0;
}
"#,
    )?)?;
    let one = buffer_create(4)?;
    compute_set(clash, "out", shader_value::ShaderValue::Buffer(one))?;
    let ambiguous = compute_set(clash, "radius", shader_value::ShaderValue::Float(9.0));
    compute_set(
        clash,
        "params.radius",
        shader_value::ShaderValue::Float(7.0),
    )?;
    compute_dispatch(clash, 1, 1, 1)?;
    let got = f32s(&buffer_read(one)?)[0];
    check(
        ambiguous.is_err() && got == 7.0,
        format!("ambiguous name rejected, params.radius set ({got})"),
    );

    Ok(ok)
}

fn main() {
    let ok = sketch().unwrap();
    println!(
        "neighbors_test: {}",
        if ok { "ALL PASS" } else { "FAILURES" }
    );
    exit(if ok { 0 } else { 1 }).unwrap();
}
