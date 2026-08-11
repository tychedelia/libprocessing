//! Stream compaction: given per-particle float keep-flags (0 = drop, non-zero =
//! keep), produce a dense buffer of the kept particle indices and return the
//! kept count. This is the write-back half of the count→scan→scatter pattern
//! (reusing [`prefix_sum_u32`]), and the basis for Delete/Group selection and a
//! dense alive set.
//!
//! Passes: flag→u32 (`compact_flag.wgsl`) → exclusive scan → scatter
//! (`compact_scatter.wgsl`). The scan buffer is length `n + 1`, so its last
//! element is the kept count (read back to the caller).

use std::sync::Mutex;

use bevy::prelude::Entity;

use processing_core::error::Result;

use crate::particles::scan::prefix_sum_u32;
use crate::shader_value::ShaderValue;
use crate::{
    buffer_create, buffer_destroy, buffer_read_element, buffer_size, compute_create,
    compute_dispatch, compute_set, shader_load,
};

const WG: u32 = 64;
const FLAG_SHADER: &str = "embedded://processing_render/particles/kernels/compact_flag.wgsl";
const SCATTER_SHADER: &str = "embedded://processing_render/particles/kernels/compact_scatter.wgsl";

static COMPUTES: Mutex<Option<(Entity, Entity)>> = Mutex::new(None);
/// The `n + 1` scan buffer, reused across calls. Kept EXACTLY sized because
/// `prefix_sum_u32` scans the whole buffer — a stale larger tail would corrupt
/// the count. Resized (destroy + recreate) only when `n` changes.
static SCANNED: Mutex<Option<(Entity, u64)>> = Mutex::new(None);

fn computes() -> Result<(Entity, Entity)> {
    let mut guard = COMPUTES.lock().unwrap();
    if let Some(v) = *guard {
        return Ok(v);
    }
    let flag = compute_create(shader_load(FLAG_SHADER)?)?;
    let scatter = compute_create(shader_load(SCATTER_SHADER)?)?;
    *guard = Some((flag, scatter));
    Ok((flag, scatter))
}

fn scanned_buffer(bytes: u64) -> Result<Entity> {
    let mut guard = SCANNED.lock().unwrap();
    if let Some((entity, size)) = *guard {
        if size == bytes {
            return Ok(entity);
        }
        let _ = buffer_destroy(entity);
    }
    let entity = buffer_create(bytes)?;
    *guard = Some((entity, bytes));
    Ok(entity)
}

/// Compact `flags` (an `array<f32>` of 0/keep values) into `indices_out` (an
/// `array<u32>` at least as long as the kept count), returning the kept count.
/// `indices_out[0..count]` are the kept particle indices, ascending.
pub fn compact(flags: Entity, indices_out: Entity) -> Result<u32> {
    let n = (buffer_size(flags)? / 4) as u32;
    if n == 0 {
        return Ok(0);
    }
    let (flag_c, scatter_c) = computes()?;
    let scanned = scanned_buffer(((n + 1) as u64) * 4)?;

    // 1. float flags -> u32 keep, trailing slot zeroed.
    compute_set(flag_c, "flags", ShaderValue::Buffer(flags))?;
    compute_set(flag_c, "scanned", ShaderValue::Buffer(scanned))?;
    compute_dispatch(flag_c, (n + 1).div_ceil(WG), 1, 1)?;

    // 2. exclusive scan: scanned[i] = dense slot for particle i; scanned[n] = count.
    prefix_sum_u32(scanned)?;

    // 3. scatter kept indices to their dense slots.
    compute_set(scatter_c, "flags", ShaderValue::Buffer(flags))?;
    compute_set(scatter_c, "scanned", ShaderValue::Buffer(scanned))?;
    compute_set(scatter_c, "indices", ShaderValue::Buffer(indices_out))?;
    compute_dispatch(scatter_c, n.div_ceil(WG), 1, 1)?;

    // Kept count = last scan element.
    let bytes = buffer_read_element(scanned, (n as u64) * 4, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}
