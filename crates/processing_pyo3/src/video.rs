use bevy::prelude::Entity;
use processing_video::{
    PlaybackMode, video_create, video_destroy, video_image, video_is_loaded, video_load,
    video_position, video_resolution, video_seek, video_set_mode, video_set_paused,
    video_set_speed,
};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::graphics::Image;

#[pyclass(unsendable)]
pub struct Video {
    entity: Entity,
}

#[pymethods]
impl Video {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let handle = video_load(path).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        let entity =
            video_create(handle).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Self { entity })
    }

    pub fn is_loaded(&self) -> PyResult<bool> {
        video_is_loaded(self.entity).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn resolution(&self) -> PyResult<(u32, u32)> {
        video_resolution(self.entity).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn image(&self) -> PyResult<Image> {
        let entity =
            video_image(self.entity).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Image::from_entity(entity))
    }

    pub fn position(&self) -> PyResult<f64> {
        video_position(self.entity).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn seek(&self, seconds: f64) -> PyResult<()> {
        video_seek(self.entity, seconds).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn play(&self) -> PyResult<()> {
        video_set_mode(self.entity, PlaybackMode::Once)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        video_set_paused(self.entity, false)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn pause(&self) -> PyResult<()> {
        video_set_paused(self.entity, true).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    #[pyo3(name = "loop")]
    pub fn loop_(&self) -> PyResult<()> {
        video_set_mode(self.entity, PlaybackMode::Loop)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        video_set_paused(self.entity, false)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn no_loop(&self) -> PyResult<()> {
        video_set_mode(self.entity, PlaybackMode::Once)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn speed(&self, rate: f32) -> PyResult<()> {
        video_set_speed(self.entity, rate).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        let _ = video_destroy(self.entity);
    }
}
