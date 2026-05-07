//! Built-in compute kernels for [`Particles`](super::Particles), embedded as
//! assets and dispatched via `particles_apply`.

use bevy::asset::embedded_asset;
use bevy::prelude::*;

pub struct ParticlesKernelsPlugin;

impl Plugin for ParticlesKernelsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "noise.wgsl");
        embedded_asset!(app, "transform.wgsl");
        embedded_asset!(app, "attr_linear.wgsl");
        embedded_asset!(app, "attr_combine.wgsl");
        embedded_asset!(app, "attr_mix.wgsl");
        embedded_asset!(app, "attr_lookup1d.wgsl");
        embedded_asset!(app, "attr_lookup2d.wgsl");
    }
}

pub const NOISE_PATH: &str = "embedded://processing_render/particles/kernels/noise.wgsl";
pub const TRANSFORM_PATH: &str =
    "embedded://processing_render/particles/kernels/transform.wgsl";
pub const ATTR_LINEAR_PATH: &str =
    "embedded://processing_render/particles/kernels/attr_linear.wgsl";
pub const ATTR_COMBINE_PATH: &str =
    "embedded://processing_render/particles/kernels/attr_combine.wgsl";
pub const ATTR_MIX_PATH: &str =
    "embedded://processing_render/particles/kernels/attr_mix.wgsl";
pub const ATTR_LOOKUP1D_PATH: &str =
    "embedded://processing_render/particles/kernels/attr_lookup1d.wgsl";
pub const ATTR_LOOKUP2D_PATH: &str =
    "embedded://processing_render/particles/kernels/attr_lookup2d.wgsl";
