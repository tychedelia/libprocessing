//! GPU-generated connectivity for the direct-rasterization path (task #31, rung
//! 2). Builds a hardware index buffer + indirect draw args from a compute kernel
//! so a particle position buffer draws as connected lines/triangles with one
//! `draw_indexed_indirect` — neither the connectivity nor the draw count ever
//! touches the CPU. The buffers are built once per topology and cached on the
//! `Particles` component; `point_render.rs` consumes them at draw time.

use std::sync::Mutex;

use bevy::prelude::Entity;
use bevy::render::render_resource::BufferUsages;

use processing_core::app_mut;
use processing_core::error;

use crate::geometry::Topology;
use crate::particles::{Connectivity, Particles};
use crate::shader_value::ShaderValue;
use crate::{
    buffer_create_with_usage, buffer_destroy, compute_create, compute_dispatch, compute_set,
    shader_load,
};

const WORKGROUP_SIZE: u32 = 64;
const SHADER: &str = "embedded://processing_render/particles/kernels/connectivity.wgsl";

/// `DrawIndexedIndirectArgs` is 5 x u32 (index_count, instance_count,
/// first_index, base_vertex, first_instance).
const INDIRECT_ARGS_BYTES: u64 = 5 * 4;

/// The `mode` uniform values understood by `connectivity.wgsl`.
const MODE_LINE_CHAIN: u32 = 0;

/// The connectivity compute pipeline, compiled once and reused.
static CONNECTIVITY_COMPUTE: Mutex<Option<Entity>> = Mutex::new(None);

fn connectivity_compute() -> error::Result<Entity> {
    let mut guard = CONNECTIVITY_COMPUTE.lock().unwrap();
    if let Some(entity) = *guard {
        return Ok(entity);
    }
    let compute = compute_create(shader_load(SHADER)?)?;
    *guard = Some(compute);
    Ok(compute)
}

/// The number of indices and the kernel `mode` a topology emits for `count`
/// particles, or an error if the topology has no built-in generator yet.
fn plan(topology: Topology, count: u32) -> error::Result<(u32, u32)> {
    match topology {
        // A polyline through consecutive particles: n-1 segments, 2 indices each.
        Topology::LineList => {
            if count < 2 {
                return Err(error::ProcessingError::InvalidArgument(
                    "connected `lines` need at least 2 particles".to_string(),
                ));
            }
            Ok(((count - 1) * 2, MODE_LINE_CHAIN))
        }
        other => Err(error::ProcessingError::InvalidArgument(format!(
            "no built-in connectivity generator for topology {other:?} yet (task #31)"
        ))),
    }
}

/// Ensure `particles` has a cached index + indirect buffer for `topology`,
/// building them on the GPU if absent or if the cached topology differs. Cheap
/// and idempotent once built, so it is safe to call every frame from the draw
/// path. Points need no connectivity — callers should skip this for `PointList`.
pub fn particles_ensure_connectivity(
    particles_entity: Entity,
    topology: Topology,
) -> error::Result<()> {
    // Fast path: already built for this topology.
    let (capacity, existing) = app_mut(|app| {
        let field = app
            .world()
            .get::<Particles>(particles_entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?;
        Ok((field.capacity, field.connectivity))
    })?;

    if let Some(existing) = existing {
        if existing.topology == topology {
            return Ok(());
        }
    }

    let (index_count, mode) = plan(topology, capacity)?;

    let index_buffer =
        buffer_create_with_usage(index_count as u64 * 4, BufferUsages::INDEX)?;
    let indirect_buffer =
        buffer_create_with_usage(INDIRECT_ARGS_BYTES, BufferUsages::INDIRECT)?;

    let compute = connectivity_compute()?;
    compute_set(compute, "indices", ShaderValue::Buffer(index_buffer))?;
    compute_set(compute, "args", ShaderValue::Buffer(indirect_buffer))?;
    compute_set(compute, "count", ShaderValue::UInt(capacity))?;
    compute_set(compute, "mode", ShaderValue::UInt(mode))?;
    compute_dispatch(compute, capacity.div_ceil(WORKGROUP_SIZE), 1, 1)?;

    // Store the new connectivity, tearing down any previous (different-topology)
    // buffers so they don't leak.
    app_mut(|app| {
        let mut field = app
            .world_mut()
            .get_mut::<Particles>(particles_entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?;
        let previous = field.connectivity.replace(Connectivity {
            topology,
            index_buffer,
            indirect_buffer,
        });
        Ok(previous)
    })
    .and_then(|previous| {
        if let Some(previous) = previous {
            buffer_destroy(previous.index_buffer)?;
            buffer_destroy(previous.indirect_buffer)?;
        }
        Ok(())
    })
}
