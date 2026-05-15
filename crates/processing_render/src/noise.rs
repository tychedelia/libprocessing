//! Entity-based noise sources backed by the `noiz` crate.

use bevy::math::{Vec2, Vec3};
use bevy::prelude::{Component, Entity, World};
use noiz::cell_noise::{
    PerCellPointDistances, WorleyDifference, WorleyLeastDistance, WorleySecondLeastDistance,
};
use noiz::cells::Voronoi;
use noiz::layering::{FractalLayers, LayeredNoise, Normed, Octave, Persistence};
use noiz::lengths::{ChebyshevLength, EuclideanLength, ManhattanLength};
use noiz::prelude::{
    Noise, SampleableFor, common_noise::Perlin, common_noise::Simplex, common_noise::Value,
};
use processing_core::error::{ProcessingError, Result};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum NoiseKind {
    /// Output in `[-1, 1]`.
    #[default]
    Perlin = 0,
    /// Output in `[-1, 1]`.
    Simplex = 1,
    /// Output in `[0, 1]`.
    Value = 2,
    /// Output in `[0, 1]`.
    Worley = 3,
}

impl From<u8> for NoiseKind {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Perlin,
            1 => Self::Simplex,
            2 => Self::Value,
            3 => Self::Worley,
            _ => Self::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum NoiseDistance {
    #[default]
    Euclidean = 0,
    Manhattan = 1,
    Chebyshev = 2,
}

impl From<u8> for NoiseDistance {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Euclidean,
            1 => Self::Manhattan,
            2 => Self::Chebyshev,
            _ => Self::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum WorleyMode {
    #[default]
    Nearest = 0,
    SecondNearest = 1,
    Difference = 2,
}

impl From<u8> for WorleyMode {
    fn from(v: u8) -> Self {
        match v {
            0 => Self::Nearest,
            1 => Self::SecondNearest,
            2 => Self::Difference,
            _ => Self::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct NoiseConfig {
    pub kind: NoiseKind,
    pub seed: u32,
    pub frequency: f32,
    pub octaves: u32,
    pub persistence: f32,
    pub lacunarity: f32,
    pub distance: NoiseDistance,
    pub worley_mode: WorleyMode,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            kind: NoiseKind::Perlin,
            seed: 0,
            frequency: 1.0,
            octaves: 4,
            persistence: 0.5,
            lacunarity: 2.0,
            distance: NoiseDistance::Euclidean,
            worley_mode: WorleyMode::Nearest,
        }
    }
}

// Concrete noiz types differ for every (kind, distance, worley_mode) combination
// and don't share a common base, so erase them behind a trait object.
trait NoiseSampler: Send + Sync {
    fn sample_2d(&self, pos: Vec2) -> f32;
    fn sample_3d(&self, pos: Vec3) -> f32;
}

#[derive(Component)]
pub struct NoiseSource {
    config: NoiseConfig,
    sampler: Box<dyn NoiseSampler>,
}

impl NoiseSource {
    fn rebuild(&mut self) {
        self.sampler = build_sampler(&self.config);
    }
}

struct Sampler<N>(Noise<N>);

impl<N> NoiseSampler for Sampler<N>
where
    Noise<N>: SampleableFor<Vec2, f32> + SampleableFor<Vec3, f32> + Send + Sync,
{
    #[inline]
    fn sample_2d(&self, pos: Vec2) -> f32 {
        SampleableFor::<Vec2, f32>::sample(&self.0, pos)
    }

    #[inline]
    fn sample_3d(&self, pos: Vec3) -> f32 {
        SampleableFor::<Vec3, f32>::sample(&self.0, pos)
    }
}

macro_rules! build_fbm {
    ($inner:expr, $config:expr) => {{
        let inner = $inner;
        let layered = LayeredNoise::new(
            Normed::<f32>::default(),
            Persistence($config.persistence),
            FractalLayers {
                layer: Octave(inner),
                lacunarity: $config.lacunarity,
                amount: $config.octaves.max(1),
            },
        );
        let mut noise = Noise::from(layered);
        noise.seed = noiz::rng::NoiseRng($config.seed);
        noise.frequency = $config.frequency;
        Box::new(Sampler(noise)) as Box<dyn NoiseSampler>
    }};
}

fn build_sampler(config: &NoiseConfig) -> Box<dyn NoiseSampler> {
    match config.kind {
        NoiseKind::Perlin => build_fbm!(Perlin::default(), config),
        NoiseKind::Simplex => build_fbm!(Simplex::default(), config),
        NoiseKind::Value => build_fbm!(Value::default(), config),
        NoiseKind::Worley => build_worley_sampler(config),
    }
}

fn build_worley_sampler(config: &NoiseConfig) -> Box<dyn NoiseSampler> {
    use NoiseDistance::*;
    use WorleyMode::*;
    match (config.distance, config.worley_mode) {
        (Euclidean, Nearest) => build_fbm!(
            PerCellPointDistances::<Voronoi, EuclideanLength, WorleyLeastDistance>::default(),
            config
        ),
        (Euclidean, SecondNearest) => build_fbm!(
            PerCellPointDistances::<Voronoi, EuclideanLength, WorleySecondLeastDistance>::default(),
            config
        ),
        (Euclidean, Difference) => build_fbm!(
            PerCellPointDistances::<Voronoi, EuclideanLength, WorleyDifference>::default(),
            config
        ),
        (Manhattan, Nearest) => build_fbm!(
            PerCellPointDistances::<Voronoi, ManhattanLength, WorleyLeastDistance>::default(),
            config
        ),
        (Manhattan, SecondNearest) => build_fbm!(
            PerCellPointDistances::<Voronoi, ManhattanLength, WorleySecondLeastDistance>::default(),
            config
        ),
        (Manhattan, Difference) => build_fbm!(
            PerCellPointDistances::<Voronoi, ManhattanLength, WorleyDifference>::default(),
            config
        ),
        (Chebyshev, Nearest) => build_fbm!(
            PerCellPointDistances::<Voronoi, ChebyshevLength, WorleyLeastDistance>::default(),
            config
        ),
        (Chebyshev, SecondNearest) => build_fbm!(
            PerCellPointDistances::<Voronoi, ChebyshevLength, WorleySecondLeastDistance>::default(),
            config
        ),
        (Chebyshev, Difference) => build_fbm!(
            PerCellPointDistances::<Voronoi, ChebyshevLength, WorleyDifference>::default(),
            config
        ),
    }
}

pub fn create(world: &mut World) -> Entity {
    let config = NoiseConfig::default();
    let sampler = build_sampler(&config);
    world.spawn(NoiseSource { config, sampler }).id()
}

pub fn destroy(world: &mut World, entity: Entity) -> Result<()> {
    if world.get::<NoiseSource>(entity).is_none() {
        return Err(ProcessingError::NoiseNotFound);
    }
    world.despawn(entity);
    Ok(())
}

fn with_mut<F>(world: &mut World, entity: Entity, f: F) -> Result<()>
where
    F: FnOnce(&mut NoiseSource),
{
    let mut source = world
        .get_mut::<NoiseSource>(entity)
        .ok_or(ProcessingError::NoiseNotFound)?;
    f(&mut source);
    source.rebuild();
    Ok(())
}

pub fn set_mode(world: &mut World, entity: Entity, kind: NoiseKind) -> Result<()> {
    with_mut(world, entity, |s| s.config.kind = kind)
}

pub fn set_seed(world: &mut World, entity: Entity, seed: u32) -> Result<()> {
    with_mut(world, entity, |s| s.config.seed = seed)
}

pub fn set_detail(
    world: &mut World,
    entity: Entity,
    octaves: u32,
    persistence: f32,
) -> Result<()> {
    with_mut(world, entity, |s| {
        s.config.octaves = octaves;
        s.config.persistence = persistence;
    })
}

pub fn set_frequency(world: &mut World, entity: Entity, freq: f32) -> Result<()> {
    with_mut(world, entity, |s| s.config.frequency = freq)
}

pub fn set_lacunarity(world: &mut World, entity: Entity, lac: f32) -> Result<()> {
    with_mut(world, entity, |s| s.config.lacunarity = lac)
}

pub fn set_distance(world: &mut World, entity: Entity, dist: NoiseDistance) -> Result<()> {
    with_mut(world, entity, |s| s.config.distance = dist)
}

pub fn set_worley_mode(world: &mut World, entity: Entity, mode: WorleyMode) -> Result<()> {
    with_mut(world, entity, |s| s.config.worley_mode = mode)
}

pub fn sample_1d(world: &World, entity: Entity, x: f32) -> Result<f32> {
    let source = world
        .get::<NoiseSource>(entity)
        .ok_or(ProcessingError::NoiseNotFound)?;
    Ok(source.sampler.sample_2d(Vec2::new(x, 0.0)))
}

pub fn sample_2d(world: &World, entity: Entity, x: f32, y: f32) -> Result<f32> {
    let source = world
        .get::<NoiseSource>(entity)
        .ok_or(ProcessingError::NoiseNotFound)?;
    Ok(source.sampler.sample_2d(Vec2::new(x, y)))
}

pub fn sample_3d(world: &World, entity: Entity, x: f32, y: f32, z: f32) -> Result<f32> {
    let source = world
        .get::<NoiseSource>(entity)
        .ok_or(ProcessingError::NoiseNotFound)?;
    Ok(source.sampler.sample_3d(Vec3::new(x, y, z)))
}
