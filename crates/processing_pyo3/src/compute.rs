use bevy::prelude::Entity;
use processing::prelude::*;
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyBytes, PyList},
};

use crate::material::py_to_material_value;
use crate::shader::Shader;

#[pyclass(unsendable)]
pub struct Buffer {
    pub(crate) entity: Entity,
    element_type: Option<shader_value::ShaderValue>,
}

#[pymethods]
impl Buffer {
    #[new]
    #[pyo3(signature = (size=None, data=None))]
    pub fn new(size: Option<u64>, data: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (entity, element_type) = if let Some(data) = data {
            let (bytes, element_type) = shader_values_to_bytes(data)?;
            let entity = buffer_create_with_data(bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
            (entity, element_type)
        } else {
            let size = size.unwrap_or(0);
            let entity =
                buffer_create(size).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
            (entity, None)
        };
        Ok(Self {
            entity,
            element_type,
        })
    }

    pub fn write(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (bytes, element_type) = shader_values_to_bytes(values)?;
        self.element_type = element_type;
        buffer_write(self.entity, bytes).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn read<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = buffer_read(self.entity).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        let Some(ref template) = self.element_type else {
            return Ok(PyBytes::new(py, &data).into_any());
        };

        let elem_size = template
            .byte_size()
            .ok_or_else(|| PyRuntimeError::new_err("unsupported element type"))?;

        let values = data
            .chunks_exact(elem_size)
            .map(|chunk| {
                let sv = template
                    .read_from_bytes(chunk)
                    .ok_or_else(|| PyRuntimeError::new_err("failed to decode bytes"))?;
                shader_value_to_py(py, &sv)
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(PyList::new(py, values)?.into_any())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = buffer_destroy(self.entity);
    }
}

fn shader_values_to_bytes(
    values: &Bound<'_, PyAny>,
) -> PyResult<(Vec<u8>, Option<shader_value::ShaderValue>)> {
    let mut bytes = Vec::new();
    let mut element_type = None;
    for item in values.try_iter()? {
        let sv = py_to_material_value(&item?)?;
        match sv.to_bytes() {
            Some(b) => {
                element_type.get_or_insert_with(|| sv.clone());
                bytes.extend_from_slice(&b);
            }
            None => return Err(PyTypeError::new_err("unsupported value type for buffer")),
        }
    }
    Ok((bytes, element_type))
}

fn shader_value_to_py<'py>(
    py: Python<'py>,
    sv: &shader_value::ShaderValue,
) -> PyResult<Bound<'py, PyAny>> {
    match sv {
        shader_value::ShaderValue::Float(v) => Ok(v.into_pyobject(py)?.into_any()),
        shader_value::ShaderValue::Float2(v) => Ok(PyList::new(py, v.iter().copied())?.into_any()),
        shader_value::ShaderValue::Float3(v) => Ok(PyList::new(py, v.iter().copied())?.into_any()),
        shader_value::ShaderValue::Float4(v) => Ok(PyList::new(py, v.iter().copied())?.into_any()),
        shader_value::ShaderValue::Int(v) => Ok(v.into_pyobject(py)?.into_any()),
        shader_value::ShaderValue::Int2(v) => Ok(PyList::new(py, v.iter().copied())?.into_any()),
        shader_value::ShaderValue::Int3(v) => Ok(PyList::new(py, v.iter().copied())?.into_any()),
        shader_value::ShaderValue::Int4(v) => Ok(PyList::new(py, v.iter().copied())?.into_any()),
        shader_value::ShaderValue::UInt(v) => Ok(v.into_pyobject(py)?.into_any()),
        shader_value::ShaderValue::Mat4(v) => Ok(PyList::new(py, v.iter().copied())?.into_any()),
        shader_value::ShaderValue::Texture(_) | shader_value::ShaderValue::Buffer(_) => Err(
            PyRuntimeError::new_err("cannot convert Texture/Buffer to Python value"),
        ),
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
        let entity =
            compute_create(shader.entity).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Self { entity })
    }

    #[pyo3(signature = (**kwargs))]
    pub fn set(&self, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<()> {
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
        compute_dispatch(self.entity, x, y, z).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

impl Drop for Compute {
    fn drop(&mut self) {
        let _ = compute_destroy(self.entity);
    }
}
