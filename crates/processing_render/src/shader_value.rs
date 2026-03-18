use bevy::prelude::*;

#[derive(Debug, Clone)]
pub enum ShaderValue {
    Float(f32),
    Float2([f32; 2]),
    Float3([f32; 3]),
    Float4([f32; 4]),
    Int(i32),
    Int2([i32; 2]),
    Int3([i32; 3]),
    Int4([i32; 4]),
    UInt(u32),
    Mat4([f32; 16]),
    Texture(Entity),
    Buffer(Entity),
}
