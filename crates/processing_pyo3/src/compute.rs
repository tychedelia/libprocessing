use bevy::prelude::Entity;
use processing::prelude::*;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use crate::material::py_to_material_value;
use crate::shader::Shader;

#[pyclass(unsendable)]
pub struct Buffer {
    pub(crate) entity: Entity,
}

#[pymethods]
impl Buffer {
    #[new]
    #[pyo3(signature = (size=None, data=None))]
    pub fn new(size: Option<u64>, data: Option<Vec<u8>>) -> PyResult<Self> {
        let entity = if let Some(data) = data {
            buffer_create_with_data(data)
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?
        } else {
            let size = size.unwrap_or(0);
            buffer_create(size).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?
        };
        Ok(Self { entity })
    }

    pub fn read(&self) -> PyResult<Vec<u8>> {
        buffer_read(self.entity).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn write(&self, data: Vec<u8>) -> PyResult<()> {
        buffer_write(self.entity, data).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = buffer_destroy(self.entity);
    }
}

#[pyclass(unsendable)]
pub struct Compute {
    pub(crate) entity: Entity,
}

#[pymethods]
impl Compute {
    #[new]
    pub fn new(shader: &Shader) -> PyResult<Self> {
        let entity = compute_create(shader.entity)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Self { entity })
    }

    #[pyo3(signature = (**kwargs))]
    pub fn set(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let Some(kwargs) = kwargs else {
            return Ok(());
        };
        for (key, value) in kwargs.iter() {
            let name: String = key.extract()?;
            let mat_value = py_to_material_value(&value)?;
            compute_set(self.entity, &name, mat_value)
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        }
        Ok(())
    }

    pub fn dispatch(&self, x: u32, y: u32, z: u32) -> PyResult<()> {
        compute_dispatch(self.entity, x, y, z)
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        let _ = compute_destroy(self.entity);
    }
}
