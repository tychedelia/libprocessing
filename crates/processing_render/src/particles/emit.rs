//! Emission and per-frame dispatch helpers for [`Particles`](super::Particles).

use bevy::prelude::*;

use processing_core::app_mut;
use processing_core::error;

use crate::geometry;
use crate::particles::kernels::KernelRequires;
use crate::particles::{Particles, particles_ensure_attribute};
use crate::shader_value::ShaderValue;
use crate::{buffer_write_element, compute_dispatch, compute_set};

const WORKGROUP_SIZE: u32 = 64;

/// GPU-driven emission into the next `count` ring-buffer slots. Auto-binds
/// attribute buffers (same convention as [`particles_apply`]) and an
/// `emit_range: vec4<f32> = (base_slot, count, capacity, 0)` uniform.
pub fn particles_emit_gpu(
    particles_entity: Entity,
    count: u32,
    compute_entity: Entity,
) -> error::Result<()> {
    if count == 0 {
        return Ok(());
    }

    let (capacity, head, buffers) = app_mut(|app| {
        let world = app.world();
        let field = world
            .get::<Particles>(particles_entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?;
        if count > field.capacity {
            return Err(error::ProcessingError::InvalidArgument(format!(
                "particles_emit_gpu count={} exceeds field capacity {}",
                count, field.capacity
            )));
        }
        let mut buffers: Vec<(String, Entity)> = Vec::with_capacity(field.buffers.len());
        for (&attr_entity, &buf_entity) in &field.buffers {
            let attr = world
                .get::<geometry::Attribute>(attr_entity)
                .ok_or(error::ProcessingError::InvalidEntity)?;
            buffers.push((attr.name.to_string(), buf_entity));
        }
        Ok((field.capacity, field.emit_head, buffers))
    })?;

    for (name, buf_entity) in buffers {
        match compute_set(compute_entity, name, ShaderValue::Buffer(buf_entity)) {
            Ok(()) => {}
            Err(error::ProcessingError::UnknownShaderProperty(_)) => {}
            Err(e) => return Err(e),
        }
    }

    for (name, value) in [
        ("emit_base", ShaderValue::UInt(head)),
        ("emit_count", ShaderValue::UInt(count)),
        ("emit_capacity", ShaderValue::UInt(capacity)),
    ] {
        match compute_set(compute_entity, name, value) {
            Ok(()) => {}
            Err(error::ProcessingError::UnknownShaderProperty(_)) => {}
            Err(e) => return Err(e),
        }
    }

    let workgroup_count = count.div_ceil(WORKGROUP_SIZE);
    compute_dispatch(compute_entity, workgroup_count, 1, 1)?;

    app_mut(|app| {
        let mut field = app
            .world_mut()
            .get_mut::<Particles>(particles_entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?;
        field.emit_head = (field.emit_head + count) % field.capacity;
        Ok(())
    })
}

/// CPU-driven emission. Writes per-attribute byte payloads into the next `n`
/// ring-buffer slots. Each entry in `attribute_data` must be exactly
/// `attr.byte_size * n` bytes. On wrap, oldest slots are overwritten.
pub fn particles_emit(
    particles_entity: Entity,
    n: u32,
    attribute_data: Vec<(Entity, Vec<u8>)>,
) -> error::Result<()> {
    if n == 0 {
        return Ok(());
    }

    let (capacity, head, attr_specs) = app_mut(|app| {
        let world = app.world();
        let field = world
            .get::<Particles>(particles_entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?;
        if n > field.capacity {
            return Err(error::ProcessingError::InvalidArgument(format!(
                "particles_emit n={} exceeds field capacity {}",
                n, field.capacity
            )));
        }
        let mut specs: Vec<(Entity, u32, Entity)> = Vec::with_capacity(attribute_data.len());
        for (attr_entity, _) in &attribute_data {
            let attr = world
                .get::<geometry::Attribute>(*attr_entity)
                .ok_or(error::ProcessingError::InvalidEntity)?;
            let buf = field.buffer(*attr_entity).ok_or_else(|| {
                error::ProcessingError::InvalidArgument(format!(
                    "particles have no buffer for attribute {:?}",
                    attr_entity
                ))
            })?;
            specs.push((*attr_entity, attr.format.byte_size() as u32, buf));
        }
        Ok((field.capacity, field.emit_head, specs))
    })?;

    for ((_, bytes), &(_, byte_size, buf)) in attribute_data.iter().zip(attr_specs.iter()) {
        let expected = (n as usize) * (byte_size as usize);
        if bytes.len() != expected {
            return Err(error::ProcessingError::InvalidArgument(format!(
                "expected {} bytes ({} particles * {} bytes), got {}",
                expected,
                n,
                byte_size,
                bytes.len()
            )));
        }
        let first_chunk_n = (capacity - head).min(n);
        let split = (first_chunk_n as usize) * (byte_size as usize);
        let first_offset = (head as u64) * (byte_size as u64);
        buffer_write_element(buf, first_offset, bytes[..split].to_vec())?;
        if first_chunk_n < n {
            buffer_write_element(buf, 0, bytes[split..].to_vec())?;
        }
    }

    app_mut(|app| {
        let mut field = app
            .world_mut()
            .get_mut::<Particles>(particles_entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?;
        field.emit_head = (field.emit_head + n) % field.capacity;
        Ok(())
    })
}

/// Dispatch a compute kernel against every slot of `particles_entity`. Auto-
/// binds each attribute buffer by name; the kernel only sees the ones it
/// declares.
pub fn particles_apply(particles_entity: Entity, compute_entity: Entity) -> error::Result<()> {
    // Lazy init: grow the system with any registered attributes this kernel's
    // manifest declares but the system doesn't yet carry. Custom computes have
    // no `KernelRequires`, so this is a no-op for them — they bind only the
    // attributes that already exist (the intersection).
    let required: Vec<Entity> = app_mut(|app| {
        Ok(app
            .world()
            .get::<KernelRequires>(compute_entity)
            .map(|r| r.0.clone())
            .unwrap_or_default())
    })?;
    for attr_entity in required {
        particles_ensure_attribute(particles_entity, attr_entity)?;
    }

    let (capacity, buffers) = app_mut(|app| {
        let world = app.world();
        let field = world
            .get::<Particles>(particles_entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?;
        let mut buffers: Vec<(String, Entity)> = Vec::with_capacity(field.buffers.len());
        for (&attr_entity, &buf_entity) in &field.buffers {
            let attr = world
                .get::<geometry::Attribute>(attr_entity)
                .ok_or(error::ProcessingError::InvalidEntity)?;
            buffers.push((attr.name.to_string(), buf_entity));
        }
        Ok((field.capacity, buffers))
    })?;

    for (name, buf_entity) in buffers {
        match compute_set(compute_entity, name, ShaderValue::Buffer(buf_entity)) {
            Ok(()) => {}
            Err(error::ProcessingError::UnknownShaderProperty(_)) => {}
            Err(e) => return Err(e),
        }
    }

    let workgroup_count = capacity.div_ceil(WORKGROUP_SIZE);
    compute_dispatch(compute_entity, workgroup_count, 1, 1)
}
