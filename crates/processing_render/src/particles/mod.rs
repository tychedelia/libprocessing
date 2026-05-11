//! gpu-resident particle / instancing container. See `docs/particles.md`.

mod emit;
pub mod kernels;
pub mod material;
pub mod pack;
mod scatter;

pub use emit::{particles_apply, particles_emit, particles_emit_gpu};
pub use kernels::{
    BOUNDS_CLAMP, BOUNDS_REFLECT, BOUNDS_SOFT, BOUNDS_WRAP, COMBINE_ADD, COMBINE_DIV, COMBINE_MAX,
    COMBINE_MIN, COMBINE_MUL, COMBINE_POW, COMBINE_SUB, FALLOFF_CONST, FALLOFF_CUBIC,
    FALLOFF_INVERSE, FALLOFF_LINEAR, FALLOFF_QUADRATIC, FALLOFF_SMOOTHSTEP,
    particles_kernel_age, particles_kernel_attr_combine, particles_kernel_attr_linear,
    particles_kernel_attr_lookup1d, particles_kernel_attr_lookup2d, particles_kernel_attr_mix,
    particles_kernel_attract, particles_kernel_bounds_box, particles_kernel_bounds_geometry,
    particles_kernel_bounds_sphere, particles_kernel_drag, particles_kernel_field,
    particles_kernel_flock, particles_kernel_force, particles_kernel_impulse,
    particles_kernel_integrate, particles_kernel_noise, particles_kernel_orient,
    particles_kernel_transform, particles_kernel_vortex,
};
pub use scatter::{
    particles_scatter_create, particles_scatter_volume_create, prepare_scatter_source,
    prepare_scatter_volume_source,
};

use bevy::asset::RenderAssetUsages;
use bevy::mesh::VertexAttributeValues;
use bevy::pbr::gpu_instance_batch::GpuInstanceBatchPlugin;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::render::RenderApp;
use bevy::render::mesh::allocator::MeshAllocatorSettings;
use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use bevy::render::renderer::RenderDevice;
use bevy::render::storage::ShaderBuffer;

use processing_core::app_mut;
use processing_core::error::{self, ProcessingError, Result};

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
    /// ring-buffer write cursor for `particles_emit`. Wraps at `capacity`.
    pub emit_head: u32,
}

impl Particles {
    pub fn buffer(&self, attribute: Entity) -> Option<Entity> {
        self.buffers.get(&attribute).copied()
    }
}

/// render-side marker pointing at the [`Particles`] entity to pack from.
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

pub fn add_attribute(
    In((particles_entity, attribute_entity, default_bytes)): In<(Entity, Entity, Vec<u8>)>,
    mut commands: Commands,
    mut particles_q: Query<&mut Particles>,
    attributes: Query<&Attribute>,
    mut shader_buffers: ResMut<Assets<ShaderBuffer>>,
    render_device: Res<RenderDevice>,
) -> Result<()> {
    let mut particles = particles_q
        .get_mut(particles_entity)
        .map_err(|_| ProcessingError::ParticlesNotFound)?;
    if particles.buffers.contains_key(&attribute_entity) {
        return Err(ProcessingError::InvalidArgument(format!(
            "particles already have attribute {attribute_entity:?}"
        )));
    }
    let attr = attributes
        .get(attribute_entity)
        .map_err(|_| ProcessingError::InvalidEntity)?;
    let elem_size = attr.format.byte_size() as usize;
    let capacity = particles.capacity as usize;
    let byte_size = capacity * elem_size;

    let initial = if default_bytes.is_empty() {
        vec![0u8; byte_size]
    } else if default_bytes.len() != elem_size {
        return Err(ProcessingError::InvalidArgument(format!(
            "default value byte size {} does not match attribute '{}' format byte size {}",
            default_bytes.len(),
            attr.name,
            elem_size,
        )));
    } else {
        let mut bytes = Vec::with_capacity(byte_size);
        for _ in 0..capacity {
            bytes.extend_from_slice(&default_bytes);
        }
        bytes
    };

    let buffer_entity = make_buffer(&mut commands, &mut shader_buffers, &render_device, &initial);
    particles.buffers.insert(attribute_entity, buffer_entity);
    Ok(())
}

pub fn particles_create(
    capacity: u32,
    attribute_entities: Vec<Entity>,
) -> error::Result<Entity> {
    app_mut(|app| {
        app.world_mut()
            .run_system_cached_with(create, (capacity, attribute_entities))
            .unwrap()
    })
}

/// capacity = `geometry`'s vertex count. Builtin attributes (`position`,
/// `normal`, `color`, `uv`) are seeded from the matching mesh attribute when
/// formats line up; everything else is zero-initialized.
pub fn particles_create_from_geometry(
    geometry_entity: Entity,
    attribute_entities: Vec<Entity>,
) -> error::Result<Entity> {
    app_mut(|app| {
        app.world_mut()
            .run_system_cached_with(create_from_geometry, (geometry_entity, attribute_entities))
            .unwrap()
    })
}

pub fn particles_destroy(entity: Entity) -> error::Result<()> {
    app_mut(|app| {
        app.world_mut()
            .run_system_cached_with(destroy, entity)
            .unwrap()
    })
}

pub fn particles_capacity(entity: Entity) -> error::Result<u32> {
    app_mut(|app| {
        Ok(app
            .world()
            .get::<Particles>(entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?
            .capacity)
    })
}

pub fn particles_buffer(
    entity: Entity,
    attribute_entity: Entity,
) -> error::Result<Option<Entity>> {
    app_mut(|app| {
        Ok(app
            .world()
            .get::<Particles>(entity)
            .ok_or(error::ProcessingError::ParticlesNotFound)?
            .buffer(attribute_entity))
    })
}

/// Add an attribute to an existing particle field, allocating its per-particle
/// buffer (sized to the field's capacity) and optionally seeding every slot
/// with `default`. Pass `None` to zero-initialize. Errors if the attribute is
/// already attached to this field, or if `default`'s type doesn't match the
/// attribute's format.
pub fn particles_attribute_add(
    particles_entity: Entity,
    attribute_entity: Entity,
    default: Option<crate::shader_value::ShaderValue>,
) -> error::Result<()> {
    let default_bytes = match default {
        None => Vec::new(),
        Some(v) => v.to_bytes().ok_or_else(|| {
            error::ProcessingError::InvalidArgument(
                "default must be a scalar/vector ShaderValue, not a Buffer/Texture/Mesh*"
                    .to_string(),
            )
        })?,
    };
    app_mut(|app| {
        app.world_mut()
            .run_system_cached_with(
                add_attribute,
                (particles_entity, attribute_entity, default_bytes),
            )
            .unwrap()
    })
}
