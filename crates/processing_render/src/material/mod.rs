pub mod pbr;

use std::collections::HashMap;

use bevy::prelude::*;

use crate::error::{ProcessingError, Result};

pub struct MaterialPlugin;

impl Plugin for MaterialPlugin {
    fn build(&self, app: &mut App) {
        // Create the default unlit material at startup
        let world = app.world_mut();
        let handle = world
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial {
                unlit: true,
                cull_mode: None,
                base_color: Color::WHITE,
                ..default()
            });
        let entity = world
            .spawn(ProcessingMaterial::Pbr(PbrMaterial {
                handle,
                overrides: HashMap::new(),
            }))
            .id();
        world.insert_resource(DefaultMaterial(entity));
    }
}

/// Resource holding the default unlit material entity.
/// Created at app init, used as the initial active material for all graphics contexts.
#[derive(Resource)]
pub struct DefaultMaterial(pub Entity);

/// A value that can be set on a material property.
#[derive(Debug, Clone)]
pub enum MaterialValue {
    Float(f32),
    Float2([f32; 2]),
    Float3([f32; 3]),
    Float4([f32; 4]),
    Int(i32),
    UInt(u32),
    Mat4([f32; 16]),
    Texture(Entity),
}

/// The processing material component, stored on material entities.
#[derive(Component)]
pub enum ProcessingMaterial {
    Pbr(Handle<StandardMaterial>),
}

pub fn create_pbr(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) -> Entity {
    let handle = materials.add(StandardMaterial {
        unlit: false,
        cull_mode: None,
        ..default()
    });
    commands.spawn(ProcessingMaterial::Pbr(handle)).id()
}

pub fn set_property(
    In((entity, name, value)): In<(Entity, String, MaterialValue)>,
    mut materials_query: Query<&mut ProcessingMaterial>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
) -> Result<()> {
    let mut mat = materials_query
        .get_mut(entity)
        .map_err(|_| ProcessingError::MaterialNotFound)?;
    match mat.as_mut() {
        ProcessingMaterial::Pbr(handle) => {
            let standard = standard_materials
                .get_mut(handle)
                .ok_or(ProcessingError::MaterialNotFound)?;
            pbr::set_property(standard, &name, &value)?;
            Ok(())
        }
    }
}

pub fn destroy(In(entity): In<Entity>, mut commands: Commands) -> Result<()> {
    commands.entity(entity).despawn();
    Ok(())
}

/// Get the StandardMaterial handle from a ProcessingMaterial entity.
pub fn resolve_standard_material_handle(
    material: &ProcessingMaterial,
) -> Option<Handle<StandardMaterial>> {
    match material {
        ProcessingMaterial::Pbr(pbr) => Some(pbr.handle.clone()),
    }
}
