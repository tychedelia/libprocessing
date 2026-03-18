use bevy::prelude::Entity;
use processing::prelude::*;
use pyo3::{
    exceptions::{PyIndexError, PyRuntimeError, PyTypeError},
    prelude::*,
    types::{PyBytes, PyList, PySlice},
};

use crate::material::py_to_material_value;
use crate::shader::Shader;

#[pyclass(unsendable)]
pub struct Buffer {
    pub(crate) entity: Entity,
    element_type: Option<shader_value::ShaderValue>,
    size: u64,
}

#[pymethods]
impl Buffer {
    #[new]
    #[pyo3(signature = (size=None, data=None))]
    pub fn new(size: Option<u64>, data: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let (entity, size, element_type) = if let Some(data) = data {
            let (bytes, element_type) = shader_values_to_bytes(data)?;
            let size = bytes.len() as u64;
            let entity = buffer_create_with_data(bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
            (entity, size, element_type)
        } else {
            let size = size.unwrap_or(0);
            let entity =
                buffer_create(size).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
            (entity, size, None)
        };
        Ok(Self {
            entity,
            element_type,
            size,
        })
    }

    pub fn __len__(&self) -> usize {
        match &self.element_type {
            Some(et) => et
                .byte_size()
                .map(|s| self.size as usize / s)
                .unwrap_or(self.size as usize),
            None => self.size as usize,
        }
    }

    pub fn __getitem__(&self, py: Python<'_>, index: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let Some(ref et) = self.element_type else {
            return Err(PyTypeError::new_err("no element type; write values first"));
        };
        let elem_size = et.byte_size().unwrap() as u64;

        if let Ok(i) = index.extract::<isize>() {
            let i = self.normalize_index(i)?;
            let bytes = buffer_read_element(self.entity, i as u64 * elem_size, elem_size)
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?
                .ok_or_else(|| {
                    PyRuntimeError::new_err("buffer data not available; call read() after dispatch")
                })?;
            let sv = et
                .read_from_bytes(&bytes)
                .ok_or_else(|| PyRuntimeError::new_err("failed to decode element"))?;
            Ok(shader_value_to_py(py, &sv)?.into())
        } else if let Ok(slice) = index.downcast::<PySlice>() {
            let len = self.__len__() as isize;
            let indices = slice.indices(len)?;
            let values = (indices.start..indices.stop)
                .step_by(indices.step as usize)
                .map(|i| {
                    let bytes = buffer_read_element(self.entity, i as u64 * elem_size, elem_size)
                        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "buffer data not available; call read() after dispatch",
                            )
                        })?;
                    let sv = et
                        .read_from_bytes(&bytes)
                        .ok_or_else(|| PyRuntimeError::new_err("failed to decode element"))?;
                    shader_value_to_py(py, &sv)
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, values)?.into())
        } else {
            Err(PyTypeError::new_err("index must be int or slice"))
        }
    }

    pub fn __setitem__(
        &mut self,
        index: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if let Ok(i) = index.extract::<isize>() {
            let sv = py_to_material_value(value)?;
            let bytes = sv
                .to_bytes()
                .ok_or_else(|| PyTypeError::new_err("unsupported value type for buffer"))?;
            let elem_size = bytes.len() as u64;
            self.element_type.get_or_insert_with(|| sv.clone());
            let i = self.normalize_index(i)?;
            buffer_write_element(self.entity, i as u64 * elem_size, bytes)
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
        } else if let Ok(slice) = index.downcast::<PySlice>() {
            let (src_bytes, element_type) = shader_values_to_bytes(value)?;
            let elem_size = element_type
                .as_ref()
                .and_then(|et| et.byte_size())
                .ok_or_else(|| PyTypeError::new_err("unsupported value type for buffer"))?
                as u64;
            if let Some(et) = element_type {
                self.element_type.get_or_insert_with(|| et);
            }
            let len = self.__len__() as isize;
            let indices = slice.indices(len)?;
            let positions: Vec<isize> = (indices.start..indices.stop)
                .step_by(indices.step as usize)
                .collect();
            let src_elems = src_bytes.len() as u64 / elem_size;
            if positions.len() as u64 != src_elems {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "slice length {} does not match value length {}",
                    positions.len(),
                    src_elems
                )));
            }
            for (pos, chunk) in positions
                .into_iter()
                .zip(src_bytes.chunks_exact(elem_size as usize))
            {
                buffer_write_element(self.entity, pos as u64 * elem_size, chunk.to_vec())
                    .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
            }
            Ok(())
        } else {
            Err(PyTypeError::new_err("index must be int or slice"))
        }
    }

    pub fn write(&mut self, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let (bytes, element_type) = shader_values_to_bytes(values)?;
        self.element_type = element_type;
        buffer_write(self.entity, bytes).map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }

    pub fn read<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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

impl Buffer {
    fn normalize_index(&self, i: isize) -> PyResult<usize> {
        let len = self.__len__() as isize;
        let i = if i < 0 { len + i } else { i };
        if i < 0 || i >= len {
            Err(PyIndexError::new_err("buffer index out of range"))
        } else {
            Ok(i as usize)
        }
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
