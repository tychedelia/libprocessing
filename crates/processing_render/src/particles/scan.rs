//! Multi-level GPU exclusive prefix-sum (scan) over an `array<u32>` buffer.
//!
//! This is the reusable primitive under spatial-hash bucketing, stream
//! compaction, and sort. It is orchestrated entirely from the CPU as a chain of
//! block-scan / block-offset-add dispatches (`scan_block.wgsl`, `scan_add.wgsl`):
//!
//! 1. Scan the input in 256-element blocks, emitting each block's total.
//! 2. Recursively scan those per-block totals (they are just another `u32`
//!    array), until a level fits in a single block and is globally exclusive.
//! 3. Walk back down, adding each level's scanned block offsets into the level
//!    below it.
//!
//! Every level derives its element count from `arrayLength`, so no uniform is
//! shared between dispatches — successive submissions on the same queue are
//! ordered by wgpu's automatic buffer hazard tracking, which is all the
//! synchronisation the data dependency needs.

use std::sync::Mutex;

use bevy::prelude::Entity;

use processing_core::error::Result;

use crate::shader_value::ShaderValue;
use crate::{
    buffer_create, buffer_destroy, buffer_size, compute_create, compute_dispatch, compute_set,
    shader_load,
};

/// Elements scanned per workgroup. Must match `@workgroup_size` and the shared
/// array length in both scan shaders.
const BLOCK: u64 = 256;

const BLOCK_SHADER: &str = "embedded://processing_render/particles/kernels/scan_block.wgsl";
const ADD_SHADER: &str = "embedded://processing_render/particles/kernels/scan_add.wgsl";

/// `(scan_block, scan_add)` compute pipelines, compiled once and reused.
static SCAN_COMPUTES: Mutex<Option<(Entity, Entity)>> = Mutex::new(None);

/// Per-level scratch block-sum buffers, keyed by level depth and reused across
/// calls. Growing a level destroys the previous buffer, which is safe: it was
/// bound in an earlier frame's submission and is idle by now.
static SCRATCH: Mutex<Vec<(Entity, u64)>> = Mutex::new(Vec::new());

fn scan_computes() -> Result<(Entity, Entity)> {
    let mut guard = SCAN_COMPUTES.lock().unwrap();
    if let Some(v) = *guard {
        return Ok(v);
    }
    let block = compute_create(shader_load(BLOCK_SHADER)?)?;
    let add = compute_create(shader_load(ADD_SHADER)?)?;
    *guard = Some((block, add));
    Ok((block, add))
}

/// A scratch block-sums buffer for `level`, at least `bytes` large.
fn scratch_buffer(level: usize, bytes: u64) -> Result<Entity> {
    let mut scratch = SCRATCH.lock().unwrap();
    while scratch.len() <= level {
        scratch.push((Entity::PLACEHOLDER, 0));
    }
    let (entity, size) = scratch[level];
    if entity != Entity::PLACEHOLDER && size >= bytes {
        return Ok(entity);
    }
    if entity != Entity::PLACEHOLDER {
        let _ = buffer_destroy(entity);
    }
    let new_entity = buffer_create(bytes)?;
    scratch[level] = (new_entity, bytes);
    Ok(new_entity)
}

/// Exclusive prefix-sum over every `u32` element of `buffer`, in place.
///
/// `buffer[i]` becomes the sum of the original `buffer[0..i]` (so `buffer[0]`
/// becomes `0`). The whole buffer is scanned; size it to exactly the element
/// count you want.
///
/// Current single-dispatch dimension limit is `65535 * 256 ≈ 16.7M` elements
/// per level; larger inputs would need dispatch tiling.
pub fn prefix_sum_u32(buffer: Entity) -> Result<()> {
    let total_bytes = buffer_size(buffer)?;
    if total_bytes == 0 {
        return Ok(());
    }

    let (block, add) = scan_computes()?;

    let n = (total_bytes / 4).max(1);

    // level_bufs[0] is the caller's buffer; each later entry holds the previous
    // level's per-block totals.
    let mut level_bufs: Vec<Entity> = vec![buffer];
    let mut level_ns: Vec<u64> = vec![n];

    let mut lvl = 0usize;
    loop {
        let num_blocks = level_ns[lvl].div_ceil(BLOCK).max(1);
        let sums = scratch_buffer(lvl, num_blocks * 4)?;

        compute_set(block, "data", ShaderValue::Buffer(level_bufs[lvl]))?;
        compute_set(block, "block_sums", ShaderValue::Buffer(sums))?;
        compute_dispatch(block, num_blocks as u32, 1, 1)?;

        level_bufs.push(sums);
        level_ns.push(num_blocks);

        if num_blocks == 1 {
            break;
        }
        lvl += 1;
    }

    // Descend, top-down: the deepest level is a single globally-exclusive block,
    // so start one above it and fold each level's offsets into the one below.
    // `level_bufs.len() - 2` is the index of that deepest scanned data buffer.
    for k in (0..level_bufs.len() - 2).rev() {
        let num_blocks = level_ns[k].div_ceil(BLOCK).max(1);
        compute_set(add, "data", ShaderValue::Buffer(level_bufs[k]))?;
        compute_set(add, "block_sums", ShaderValue::Buffer(level_bufs[k + 1]))?;
        compute_dispatch(add, num_blocks as u32, 1, 1)?;
    }

    Ok(())
}
