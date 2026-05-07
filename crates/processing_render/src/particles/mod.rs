//! GPU-resident particle / instancing container. See `docs/particles.md`.

pub mod kernels;
pub mod material;
pub mod pack;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, VertexAttributeValues};
use bevy::pbr::gpu_instance_batch::GpuInstanceBatchPlugin;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::mesh::allocator::MeshAllocatorSettings;
use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::storage::ShaderBuffer;

use processing_core::error::{ProcessingError, Result};

use crate::compute;
use crate::geometry::{Attribute, AttributeFormat, Geometry};

pub struct ParticlesPlugin;

impl Plugin for ParticlesPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(GpuInstanceBatchPlugin);
        app.add_plugins(pack::ParticlesPackPlugin);
        app.add_plugins(material::ParticlesMaterialPlugin);
        app.add_plugins(kernels::ParticlesKernelsPlugin);
    }

    fn finish(&self, app: &mut App) {
        // mesh attribute/index buffers feed compute kernels (scatter, vertex
        // displacement). the mesh allocator emits its gpu buffers before the
        // render device exists, so the STORAGE usage flag has to be flipped
        // in finish() rather than build().
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .world_mut()
            .resource_mut::<MeshAllocatorSettings>()
            .extra_buffer_usages |= BufferUsages::STORAGE;
    }
}

#[derive(Component)]
pub struct Particles {
    pub capacity: u32,
    /// `Attribute` entity to backing `compute::Buffer` entity.
    pub buffers: HashMap<Entity, Entity>,
    /// Persistent rasterization entity. Must outlive the per-frame draw:
    /// `GpuInstanceBatchReservations` queues mesh batches one frame behind,
    /// so respawning per-frame loses the reservation.
    pub draw_entity: Option<Entity>,
    /// Ring-buffer write cursor for `particles_emit`. Wraps at `capacity`.
    pub emit_head: u32,
}

impl Particles {
    pub fn buffer(&self, attribute: Entity) -> Option<Entity> {
        self.buffers.get(&attribute).copied()
    }
}

/// Render-side marker pointing at the [`Particles`] entity to pack from.
#[derive(Component, Clone, Copy)]
pub struct ParticlesDraw {
    pub particles: Entity,
}

pub fn create(
    In((capacity, attribute_entities)): In<(u32, Vec<Entity>)>,
    mut commands: Commands,
    attributes: Query<&Attribute>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    render_device: Res<RenderDevice>,
) -> Result<Entity> {
    let mut buffers = HashMap::with_capacity(attribute_entities.len());
    for attr_entity in attribute_entities {
        let attr = attributes
            .get(attr_entity)
            .map_err(|_| ProcessingError::InvalidEntity)?;
        let byte_size = capacity as u64 * attr.format.byte_size() as u64;
        let buffer_entity = make_buffer(
            &mut commands,
            &mut shader_buffers,
            &render_device,
            &vec![0u8; byte_size as usize],
        );
        buffers.insert(attr_entity, buffer_entity);
    }

    let entity = commands
        .spawn(Particles {
            capacity,
            buffers,
            draw_entity: None,
            emit_head: 0,
        })
        .id();
    Ok(entity)
}

/// Capacity is the source mesh's vertex count. Registered attributes are
/// seeded from the matching mesh attribute (by name + format); unmatched
/// ones are zero-initialized.
pub fn create_from_geometry(
    In((geom_entity, attribute_entities)): In<(Entity, Vec<Entity>)>,
    mut commands: Commands,
    geometries: Query<&Geometry>,
    attributes: Query<&Attribute>,
    meshes: Res<Assets<Mesh>>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    render_device: Res<RenderDevice>,
) -> Result<Entity> {
    let geom = geometries
        .get(geom_entity)
        .map_err(|_| ProcessingError::GeometryNotFound)?;
    let mesh = meshes
        .get(&geom.handle)
        .ok_or(ProcessingError::GeometryNotFound)?;
    let capacity = mesh.count_vertices() as u32;

    let mut buffers = HashMap::with_capacity(attribute_entities.len());
    for attr_entity in attribute_entities {
        let attr = attributes
            .get(attr_entity)
            .map_err(|_| ProcessingError::InvalidEntity)?;
        let byte_size = capacity as u64 * attr.format.byte_size() as u64;

        let initial = mesh
            .attribute(attr.inner)
            .and_then(|values| attribute_values_to_bytes(values, attr.format))
            .filter(|bytes| bytes.len() == byte_size as usize)
            .unwrap_or_else(|| vec![0u8; byte_size as usize]);

        let buffer_entity =
            make_buffer(&mut commands, &mut shader_buffers, &render_device, &initial);
        buffers.insert(attr_entity, buffer_entity);
    }

    let entity = commands
        .spawn(Particles {
            capacity,
            buffers,
            draw_entity: None,
            emit_head: 0,
        })
        .id();
    Ok(entity)
}

fn make_buffer(
    commands: &mut Commands,
    shader_buffers: &mut Assets<ShaderBuffer>,
    render_device: &RenderDevice,
    initial: &[u8],
) -> Entity {
    let byte_size = initial.len() as u64;
    let handle = shader_buffers.add(ShaderBuffer::new(initial, RenderAssetUsages::all()));
    let readback = render_device.create_buffer(&BufferDescriptor {
        label: Some("Particles Buffer Readback"),
        size: byte_size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    commands
        .spawn(compute::Buffer {
            handle,
            readback_buffer: readback,
            size: byte_size,
            synced: true,
            bound_rw: false,
        })
        .id()
}

fn attribute_values_to_bytes(
    values: &VertexAttributeValues,
    format: AttributeFormat,
) -> Option<Vec<u8>> {
    match (format, values) {
        (AttributeFormat::Float, VertexAttributeValues::Float32(v)) => {
            Some(v.iter().flat_map(|f| f.to_le_bytes()).collect())
        }
        (AttributeFormat::Float2, VertexAttributeValues::Float32x2(v)) => Some(
            v.iter()
                .flat_map(|p| p.iter().flat_map(|f| f.to_le_bytes()))
                .collect(),
        ),
        (AttributeFormat::Float3, VertexAttributeValues::Float32x3(v)) => Some(
            v.iter()
                .flat_map(|p| p.iter().flat_map(|f| f.to_le_bytes()))
                .collect(),
        ),
        (AttributeFormat::Float4, VertexAttributeValues::Float32x4(v)) => Some(
            v.iter()
                .flat_map(|p| p.iter().flat_map(|f| f.to_le_bytes()))
                .collect(),
        ),
        _ => None,
    }
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

/// CPU side of `particles_scatter_create`. Deinterleaves the source mesh so
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

/// CPU side of `particles_scatter_volume_create`. Like
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

pub fn destroy(
    In(entity): In<Entity>,
    mut commands: Commands,
    particles: Query<&Particles>,
) -> Result<()> {
    let p = particles
        .get(entity)
        .map_err(|_| ProcessingError::ParticlesNotFound)?;
    for &buffer_entity in p.buffers.values() {
        commands.entity(buffer_entity).despawn();
    }
    if let Some(draw_entity) = p.draw_entity {
        commands.entity(draw_entity).despawn();
    }
    commands.entity(entity).despawn();
    Ok(())
}
