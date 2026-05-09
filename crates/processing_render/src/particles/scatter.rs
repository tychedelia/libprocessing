//! Sprinkle-style scatter kernels: build a [`Compute`](crate::compute::Compute)
//! kernel that emits particles uniformly across a source mesh's surface or
//! volume, dispatched via [`particles_emit_gpu`](super::particles_emit_gpu).

use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::prelude::*;

use processing_core::app_mut;
use processing_core::error::{self, ProcessingError, Result};

use crate::geometry::Geometry;
use crate::shader_value::ShaderValue;
use crate::{
    buffer_create_with_data, compute_create, compute_set, geometry_attribute_position, shader_load,
};

/// Build a Sprinkle-style surface scatter kernel from a source [`Geometry`].
///
/// Mutates the source mesh asset to use per-attribute vertex bindings (so the
/// position buffer can be bound to compute), and uploads a face-area CDF and a
/// dense `u32` index buffer. Returns a [`Compute`](crate::compute::Compute)
/// kernel pre-configured for [`particles_emit_gpu`](super::particles_emit_gpu).
///
/// Subsequent renders of the same mesh use split bindings; if you also want
/// the mesh rendered with interleaved vertex layout, clone the asset before
/// passing it here.
///
/// ```ignore
/// let source = geometry_torus(1.0, 0.4, 32, 16)?;
/// let scatter = particles_scatter_create(source)?;
/// // Each frame:
/// particles_emit_gpu(p, n_per_frame, scatter)?;
/// ```
pub fn particles_scatter_create(source_geometry: Entity) -> error::Result<Entity> {
    let (cdf_bytes, indices_bytes, face_count) = app_mut(|app| {
        app.world_mut()
            .run_system_cached_with(prepare_scatter_source, source_geometry)
            .unwrap()
    })?;

    let cdf_buf = buffer_create_with_data(cdf_bytes)?;
    let idx_buf = buffer_create_with_data(indices_bytes)?;

    let shader =
        shader_load("embedded://processing_render/particles/kernels/scatter_surface.wgsl")?;
    let scatter = compute_create(shader)?;

    let position_attr = geometry_attribute_position();
    compute_set(
        scatter,
        "source_position",
        ShaderValue::MeshAttribute(source_geometry, position_attr),
    )?;
    compute_set(
        scatter,
        "source_indices",
        ShaderValue::Buffer(idx_buf),
    )?;
    compute_set(scatter, "cdf", ShaderValue::Buffer(cdf_buf))?;
    compute_set(scatter, "face_count", ShaderValue::UInt(face_count))?;
    // non-zero default; override via `compute_set(_, "seed", ...)`
    compute_set(scatter, "seed", ShaderValue::UInt(0xc0ffeeu32))?;

    Ok(scatter)
}

/// Sprinkle "Volume" mode analogue. Builds a scatter kernel that emits
/// particles uniformly inside the source mesh's volume by AABB rejection
/// sampling: pick a random point in the AABB, ray-cast through the mesh, and
/// keep it if the parity (odd hits) says it's inside. Up to `max_attempts`
/// retries per particle (set via `compute_set(_, "max_attempts", UInt(N))`;
/// default 32).
///
/// The mesh must be closed for the parity test to be reliable. Cost scales
/// as `O(face_count × max_attempts)` per emitted particle.
///
/// Like [`particles_scatter_create`], deinterleaves the source mesh asset.
pub fn particles_scatter_volume_create(source_geometry: Entity) -> error::Result<Entity> {
    let (indices_bytes, aabb_min, aabb_max, face_count) = app_mut(|app| {
        app.world_mut()
            .run_system_cached_with(prepare_scatter_volume_source, source_geometry)
            .unwrap()
    })?;

    let idx_buf = buffer_create_with_data(indices_bytes)?;

    let shader =
        shader_load("embedded://processing_render/particles/kernels/scatter_volume.wgsl")?;
    let scatter = compute_create(shader)?;

    let position_attr = geometry_attribute_position();
    compute_set(
        scatter,
        "source_position",
        ShaderValue::MeshAttribute(source_geometry, position_attr),
    )?;
    compute_set(
        scatter,
        "source_indices",
        ShaderValue::Buffer(idx_buf),
    )?;
    compute_set(
        scatter,
        "aabb_min",
        ShaderValue::Float4([aabb_min[0], aabb_min[1], aabb_min[2], 0.0]),
    )?;
    compute_set(
        scatter,
        "aabb_max",
        ShaderValue::Float4([aabb_max[0], aabb_max[1], aabb_max[2], 0.0]),
    )?;
    compute_set(scatter, "face_count", ShaderValue::UInt(face_count))?;
    compute_set(scatter, "max_attempts", ShaderValue::UInt(32))?;
    compute_set(scatter, "seed", ShaderValue::UInt(0xc0ffeeu32))?;

    Ok(scatter)
}

// deinterleaves the mesh and pulls out positions + a dense u32 index list.
// the dense index list sidesteps the mesh allocator's slab offset and any
// u16 index format.
fn extract_scatter_geometry(mesh: &mut Mesh) -> Result<(Vec<[f32; 3]>, Vec<u32>)> {
    mesh.deinterleave();

    let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(p)) => p.clone(),
        _ => {
            return Err(ProcessingError::InvalidArgument(
                "scatter source mesh has no Float32x3 position attribute".to_string(),
            ));
        }
    };

    let dense_indices: Vec<u32> = match mesh.indices() {
        Some(Indices::U16(v)) => v.iter().map(|&i| i as u32).collect(),
        Some(Indices::U32(v)) => v.clone(),
        None => {
            if positions.len() % 3 != 0 {
                return Err(ProcessingError::InvalidArgument(
                    "scatter source mesh has no indices and a vertex count that isn't a \
                     multiple of 3"
                        .to_string(),
                ));
            }
            (0..positions.len() as u32).collect()
        }
    };

    if dense_indices.len() % 3 != 0 {
        return Err(ProcessingError::InvalidArgument(
            "scatter source mesh has a non-triangle index list".to_string(),
        ));
    }
    if dense_indices.is_empty() {
        return Err(ProcessingError::InvalidArgument(
            "scatter source mesh has no triangles".to_string(),
        ));
    }

    Ok((positions, dense_indices))
}

/// CPU side of [`particles_scatter_create`]. Deinterleaves the source mesh so
/// its position attribute becomes its own GPU buffer, then builds a normalized
/// face-area CDF and a dense `u32` index list for area-weighted triangle
/// sampling on the GPU.
///
/// Returns `(cdf_bytes, indices_bytes, face_count)`.
pub fn prepare_scatter_source(
    In(geom_entity): In<Entity>,
    geometries: Query<&Geometry>,
    mut meshes: ResMut<Assets<Mesh>>,
) -> Result<(Vec<u8>, Vec<u8>, u32)> {
    let geom = geometries
        .get(geom_entity)
        .map_err(|_| ProcessingError::GeometryNotFound)?;
    let mesh = meshes
        .get_mut(&geom.handle)
        .ok_or(ProcessingError::GeometryNotFound)?
        .into_inner();

    let (positions, dense_indices) = extract_scatter_geometry(mesh)?;
    let face_count = (dense_indices.len() / 3) as u32;

    let mut cum = Vec::with_capacity(face_count as usize);
    let mut total = 0.0_f32;
    for face in 0..face_count as usize {
        let i0 = dense_indices[face * 3] as usize;
        let i1 = dense_indices[face * 3 + 1] as usize;
        let i2 = dense_indices[face * 3 + 2] as usize;
        let p0 = Vec3::from_array(positions[i0]);
        let p1 = Vec3::from_array(positions[i1]);
        let p2 = Vec3::from_array(positions[i2]);
        let area = 0.5 * (p1 - p0).cross(p2 - p0).length();
        total += area;
        cum.push(total);
    }
    if total <= 0.0 {
        return Err(ProcessingError::InvalidArgument(
            "scatter source mesh has zero surface area".to_string(),
        ));
    }
    let inv = 1.0 / total;
    for v in &mut cum {
        *v *= inv;
    }

    let cdf_bytes: Vec<u8> = cum.iter().flat_map(|f| f.to_le_bytes()).collect();
    let indices_bytes: Vec<u8> = dense_indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    Ok((cdf_bytes, indices_bytes, face_count))
}

/// CPU side of [`particles_scatter_volume_create`]. Like
/// [`prepare_scatter_source`] but returns the AABB instead of a CDF; volume
/// scatter samples uniformly inside the AABB and rejection-tests against the
/// mesh on the GPU.
///
/// Returns `(indices_bytes, aabb_min, aabb_max, face_count)`.
pub fn prepare_scatter_volume_source(
    In(geom_entity): In<Entity>,
    geometries: Query<&Geometry>,
    mut meshes: ResMut<Assets<Mesh>>,
) -> Result<(Vec<u8>, [f32; 3], [f32; 3], u32)> {
    let geom = geometries
        .get(geom_entity)
        .map_err(|_| ProcessingError::GeometryNotFound)?;
    let mesh = meshes
        .get_mut(&geom.handle)
        .ok_or(ProcessingError::GeometryNotFound)?
        .into_inner();

    let (positions, dense_indices) = extract_scatter_geometry(mesh)?;
    let face_count = (dense_indices.len() / 3) as u32;

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for p in &positions {
        let v = Vec3::from_array(*p);
        min = min.min(v);
        max = max.max(v);
    }

    let indices_bytes: Vec<u8> = dense_indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    Ok((indices_bytes, min.to_array(), max.to_array(), face_count))
}
