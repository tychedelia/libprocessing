use bevy::prelude::Entity;
use processing::prelude::*;
use processing_render::noise;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

#[pyclass(unsendable)]
pub struct Noise {
    pub(crate) entity: Entity,
}

#[pymethods]
impl Noise {
    #[new]
    pub fn new() -> PyResult<Self> {
        let entity = noise_create().map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Self { entity })
    }

    pub fn mode(&self, kind: u8) -> PyResult<()> {
        noise_mode(self.entity, noise::NoiseKind::from(kind))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn seed(&self, seed: u32) -> PyResult<()> {
        noise_seed(self.entity, seed).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn detail(&self, octaves: u32, persistence: f32) -> PyResult<()> {
        noise_detail(self.entity, octaves, persistence)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn frequency(&self, freq: f32) -> PyResult<()> {
        noise_frequency(self.entity, freq).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn lacunarity(&self, lac: f32) -> PyResult<()> {
        noise_lacunarity(self.entity, lac).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn distance(&self, dist: u8) -> PyResult<()> {
        noise_distance(self.entity, noise::NoiseDistance::from(dist))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn worley(&self, mode: u8) -> PyResult<()> {
        noise_worley(self.entity, noise::WorleyMode::from(mode))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (x, y=None, z=None))]
    pub fn sample(&self, x: f32, y: Option<f32>, z: Option<f32>) -> PyResult<f32> {
        let result = match (y, z) {
            (None, _) => noise_sample_1d(self.entity, x),
            (Some(y), None) => noise_sample(self.entity, x, y),
            (Some(y), Some(z)) => noise_sample_3d(self.entity, x, y, z),
        };
        result.map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

impl Drop for Noise {
    fn drop(&mut self) {
        let _ = noise_destroy(self.entity);
    }
}
