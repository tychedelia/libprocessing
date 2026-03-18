use std::ops::Deref;

use bevy::{
    ecs::entity::EntityHashMap,
    prelude::*,
    render::{
        ExtractSchedule, MainWorld,
        render_asset::{AssetExtractionSystems, RenderAssets},
        render_resource::{
            BindGroupEntry, Buffer as WgpuBuffer, BufferDescriptor, BufferInitDescriptor,
            BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, MapMode,
            PipelineLayoutDescriptor, PollType, RawComputePipelineDescriptor,
            ShaderModuleDescriptor, ShaderSource,
        },
        renderer::{RenderDevice, RenderQueue},
        storage::{GpuShaderBuffer, ShaderBuffer},
    },
};
use bevy::asset::RenderAssetUsages;
use bevy::reflect::{PartialReflect, ReflectMut};
use bevy::shader::Shader as ShaderAsset;

use bevy_naga_reflect::dynamic_shader::DynamicShader;
use bevy_naga_reflect::reflect::ParameterCategory;

use crate::material::custom::{Shader, find_param_containing_field, shader_value_to_reflect};
use crate::shader_value::ShaderValue;
use processing_core::error::{ProcessingError, Result};

#[derive(Component)]
pub struct Buffer {
    pub handle: Handle<ShaderBuffer>,
    readback_buffer: WgpuBuffer,
    pub size: u64,
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct BufferGpuBuffers(EntityHashMap<WgpuBuffer>);

fn sync_buffers(
    mut main_world: ResMut<MainWorld>,
    gpu_buffers: Res<RenderAssets<GpuShaderBuffer>>,
) {
    main_world.resource_scope(|world, mut buffer_gpu_buffers: Mut<BufferGpuBuffers>| {
        let mut buffers = world.query::<(Entity, &Buffer)>();
        for (entity, buffer) in buffers.iter(world) {
            if let Some(gpu_buffer) = gpu_buffers.get(&buffer.handle) {
                buffer_gpu_buffers.insert(entity, gpu_buffer.buffer.clone());
            }
        }
    });
}

pub fn create_buffer(
    In(size): In<u64>,
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    render_device: Res<RenderDevice>,
) -> Entity {
    let shader_buffer = ShaderBuffer::with_size(size as usize, RenderAssetUsages::all());
    let handle = buffers.add(shader_buffer);

    let readback_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Buffer Readback"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    commands
        .spawn(Buffer {
            handle,
            readback_buffer,
            size,
        })
        .id()
}

pub fn create_buffer_with_data(
    In(data): In<Vec<u8>>,
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    render_device: Res<RenderDevice>,
) -> Entity {
    let size = data.len() as u64;
    let shader_buffer = ShaderBuffer::new(&data, RenderAssetUsages::all());
    let handle = buffers.add(shader_buffer);

    let readback_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("Buffer Readback"),
        size,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    commands
        .spawn(Buffer {
            handle,
            readback_buffer,
            size,
        })
        .id()
}

pub fn write_buffer(
    In((entity, data)): In<(Entity, Vec<u8>)>,
    p_buffers: Query<&Buffer>,
    buffer_gpu_buffers: Res<BufferGpuBuffers>,
    render_queue: Res<RenderQueue>,
) -> Result<()> {
    let _p_buffer = p_buffers
        .get(entity)
        .map_err(|_| ProcessingError::BufferNotFound)?;
    let gpu_buffer = buffer_gpu_buffers
        .get(&entity)
        .ok_or(ProcessingError::BufferNotFound)?;

    render_queue.write_buffer(gpu_buffer, 0, &data);
    Ok(())
}

pub fn read_buffer(
    In(entity): In<Entity>,
    p_buffers: Query<&Buffer>,
    buffer_gpu_buffers: Res<BufferGpuBuffers>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) -> Result<Vec<u8>> {
    let p_buffer = p_buffers
        .get(entity)
        .map_err(|_| ProcessingError::BufferNotFound)?;
    let gpu_buffer = buffer_gpu_buffers
        .get(&entity)
        .ok_or(ProcessingError::BufferNotFound)?;

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(gpu_buffer, 0, &p_buffer.readback_buffer, 0, p_buffer.size);
    render_queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = p_buffer.readback_buffer.slice(..);
    let (s, r) = crossbeam_channel::bounded(1);
    buffer_slice.map_async(MapMode::Read, move |result| match result {
        Ok(r) => s.send(r).expect("Failed to send map update"),
        Err(err) => panic!("Failed to map buffer {err}"),
    });
    render_device
        .poll(PollType::wait_indefinitely())
        .expect("Failed to poll device for map async");
    r.recv().expect("Failed to receive the map_async message");

    let data = buffer_slice.get_mapped_range().to_vec();
    p_buffer.readback_buffer.unmap();

    Ok(data)
}

pub fn destroy_buffer(
    In(entity): In<Entity>,
    mut commands: Commands,
    p_buffers: Query<&Buffer>,
    mut buffer_gpu_buffers: ResMut<BufferGpuBuffers>,
) -> Result<()> {
    p_buffers
        .get(entity)
        .map_err(|_| ProcessingError::BufferNotFound)?;
    buffer_gpu_buffers.remove(&entity);
    commands.entity(entity).despawn();
    Ok(())
}

#[derive(Component, Clone)]
pub struct Compute {
    pub shader: DynamicShader,
    pub shader_handle: Handle<ShaderAsset>,
    pub wgsl: String,
    pub workgroup_size: [u32; 3],
    pub entry_point: String,
}

pub fn create_compute(
    In(shader_entity): In<Entity>,
    mut commands: Commands,
    shader_programs: Query<&Shader>,
) -> Result<Entity> {
    let program = shader_programs
        .get(shader_entity)
        .map_err(|_| ProcessingError::ShaderNotFound)?;

    let compute_ep = program
        .module
        .entry_points
        .iter()
        .find(|ep| ep.stage == naga::ShaderStage::Compute)
        .ok_or_else(|| {
            ProcessingError::ShaderCompilationError(
                "Shader has no @compute entry point".to_string(),
            )
        })?;
    let workgroup_size = compute_ep.workgroup_size;
    let entry_point = compute_ep.name.clone();

    let mut shader = DynamicShader::new(program.module.clone())
        .map_err(|e| ProcessingError::ShaderCompilationError(e.to_string()))?;
    shader.init();

    let wgsl = {
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&program.module)
            .map_err(|e| ProcessingError::ShaderCompilationError(e.to_string()))?;
        let mut wgsl_out = String::new();
        let mut writer = naga::back::wgsl::Writer::new(
            &mut wgsl_out,
            naga::back::wgsl::WriterFlags::empty(),
        );
        writer
            .write(&program.module, &info)
            .map_err(|e| ProcessingError::ShaderCompilationError(e.to_string()))?;
        wgsl_out
    };

    Ok(commands
        .spawn(Compute {
            shader,
            shader_handle: program.shader_handle.clone(),
            workgroup_size,
            entry_point,
            wgsl,
        })
        .id())
}

pub fn set_compute_property(
    In((entity, name, value)): In<(Entity, String, ShaderValue)>,
    mut computes: Query<&mut Compute>,
    p_buffers: Query<&Buffer>,
) -> Result<()> {
    let mut compute = computes
        .get_mut(entity)
        .map_err(|_| ProcessingError::ComputeNotFound)?;

    if let ShaderValue::Buffer(buf_entity) = value {
        let buffer = p_buffers
            .get(buf_entity)
            .map_err(|_| ProcessingError::BufferNotFound)?;
        compute.shader.insert(&name, buffer.handle.clone());
        return Ok(());
    }

    let reflect_value: Box<dyn PartialReflect> = shader_value_to_reflect(&value)?;

    if let Some(field) = compute.shader.field_mut(&name) {
        field.apply(&*reflect_value);
        return Ok(());
    }

    let param_name = find_param_containing_field(&compute.shader, &name);
    if let Some(param_name) = param_name
        && let Some(param) = compute.shader.field_mut(&param_name)
        && let ReflectMut::Struct(s) = param.reflect_mut()
        && let Some(field) = s.field_mut(&name)
    {
        field.apply(&*reflect_value);
        return Ok(());
    }

    Err(ProcessingError::UnknownMaterialProperty(name))
}

enum ExtractedBinding {
    Storage(Handle<ShaderBuffer>),
    Uniform(Vec<u8>),
}

pub fn dispatch(
    In((entity, x, y, z)): In<(Entity, u32, u32, u32)>,
    world: &mut World,
) -> Result<()> {
    let (wgsl, entry_point, layout_entries_per_group, bindings_per_group): (
        String,
        String,
        Vec<(u32, Vec<bevy::render::render_resource::BindGroupLayoutEntry>)>,
        Vec<(u32, Vec<(u32, ExtractedBinding)>)>,
    ) = {
        let compute = world
            .get::<Compute>(entity)
            .ok_or(ProcessingError::ComputeNotFound)?;
        let reflection = compute.shader.reflection();

        let groups: std::collections::BTreeSet<u32> =
            reflection.parameters().map(|p| p.group()).collect();

        let mut layout_entries_per_group = Vec::new();
        let mut bindings_per_group = Vec::new();

        for &group in &groups {
            let layout_entries = reflection.bind_group_layout(group);
            let mut bindings: Vec<(u32, ExtractedBinding)> = Vec::new();

            for param in reflection.parameters().filter(|p| p.group() == group) {
                let Some(name) = param.name() else { continue };
                let binding_index = param.binding();

                match param.category() {
                    ParameterCategory::Storage { .. } => {
                        let handle = compute
                            .shader
                            .get::<Handle<ShaderBuffer>>(name)
                            .ok_or(ProcessingError::BufferNotFound)?
                            .clone();
                        bindings.push((binding_index, ExtractedBinding::Storage(handle)));
                    }
                    ParameterCategory::Uniform => {
                        let Some(field_value) =
                            bevy_naga_reflect::binding::find_field(&compute.shader, name)
                        else {
                            continue;
                        };
                        let ty = &reflection.module().types
                            [reflection.module().global_variables[param.var_handle()].ty];
                        let mut buffer =
                            bevy::render::render_resource::encase::UniformBuffer::new(Vec::new());
                        bevy_naga_reflect::binding::write_to_buffer(
                            field_value,
                            reflection.module(),
                            ty,
                            &mut buffer,
                        );
                        bindings.push((binding_index, ExtractedBinding::Uniform(buffer.as_ref().to_vec())));
                    }
                    _ => continue,
                }
            }

            layout_entries_per_group.push((group, layout_entries));
            bindings_per_group.push((group, bindings));
        }

        (
            compute.wgsl.clone(),
            compute.entry_point.clone(),
            layout_entries_per_group,
            bindings_per_group,
        )
    };

    let render_device = world.resource::<RenderDevice>();
    let shader_module = render_device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("compute_shader"),
        source: ShaderSource::Wgsl(wgsl.as_str().into()),
    });

    let mut bind_group_layouts = Vec::new();
    let mut bind_groups = Vec::new();

    for ((_, layout_entries), (_, bindings)) in
        layout_entries_per_group.iter().zip(bindings_per_group.iter())
    {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("compute_bind_group_layout"),
            layout_entries,
        );

        let mut entries: Vec<(u32, WgpuBuffer)> = Vec::new();
        for (binding_index, extracted) in bindings {
            match extracted {
                ExtractedBinding::Storage(handle) => {
                    let mut buffer_query = world.query::<(Entity, &Buffer)>();
                    let buf_entity = buffer_query
                        .iter(world)
                        .find(|(_, buf)| buf.handle.id() == handle.id())
                        .map(|(e, _)| e)
                        .ok_or(ProcessingError::BufferNotFound)?;
                    let buffer_gpu_buffers = world.resource::<BufferGpuBuffers>();
                    let gpu_buffer = buffer_gpu_buffers
                        .get(&buf_entity)
                        .ok_or(ProcessingError::BufferNotFound)?
                        .clone();
                    entries.push((*binding_index, gpu_buffer));
                }
                ExtractedBinding::Uniform(bytes) => {
                    let render_device = world.resource::<RenderDevice>();
                    let gpu_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
                        label: None,
                        usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                        contents: bytes.as_ref(),
                    });
                    entries.push((*binding_index, gpu_buffer));
                }
            }
        }

        let bind_group_entries: Vec<BindGroupEntry> = entries
            .iter()
            .map(|(binding, buf)| BindGroupEntry {
                binding: *binding,
                resource: buf.as_entire_binding(),
            })
            .collect();

        let render_device = world.resource::<RenderDevice>();
        let bind_group = render_device.create_bind_group(
            Some("compute_bind_group"),
            &layout,
            &bind_group_entries,
        );
        bind_group_layouts.push(layout);
        bind_groups.push(bind_group);
    }

    let render_device = world.resource::<RenderDevice>();
    let wgpu_layouts: Vec<_> = bind_group_layouts.iter().map(|l| l.deref()).collect();
    let pipeline_layout = render_device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("compute_pipeline_layout"),
        bind_group_layouts: &wgpu_layouts,
        immediate_size: 0,
    });

    let pipeline = render_device.create_compute_pipeline(&RawComputePipelineDescriptor {
        label: Some("compute_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: Some(&entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let render_device = world.resource::<RenderDevice>();
    let render_queue = world.resource::<RenderQueue>();
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("compute_pass"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        for (i, bg) in bind_groups.iter().enumerate() {
            pass.set_bind_group(i as u32, bg, &[]);
        }
        pass.dispatch_workgroups(x, y, z);
    }
    render_queue.submit(std::iter::once(encoder.finish()));

    Ok(())
}

pub fn destroy_compute(In(entity): In<Entity>, mut commands: Commands) -> Result<()> {
    commands.entity(entity).despawn();
    Ok(())
}

pub struct ComputePlugin;

impl Plugin for ComputePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<BufferGpuBuffers>();

        let render_app = app.sub_app_mut(bevy::render::RenderApp);
        render_app.add_systems(ExtractSchedule, sync_buffers.after(AssetExtractionSystems));
    }
}
