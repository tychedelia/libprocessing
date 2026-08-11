// Shared helpers for the particle compute kernels, imported via
// `import processing::particles::{...}`. Centralises logic that was copy-pasted
// across kernels (the falloff ladder; the grid cell math).

// Region-of-influence falloff at distance `d` within `radius`, selected by
// `mode` (matches the Rust FALLOFF_* constants):
//   0 const · 1 linear · 2 smoothstep · 3 quadratic · 4 cubic · 5 inverse
// Callers gate on `d < radius` themselves; this assumes `d <= radius` for
// modes 0-4 (n >= 0).
fn falloff(d: f32, radius: f32, mode: u32) -> f32 {
    let n = 1.0 - d / radius;
    switch mode {
        case 1u: { return n; }
        case 2u: { return n * n * (3.0 - 2.0 * n); }
        case 3u: { return n * n; }
        case 4u: { return n * n * n; }
        case 5u: { return radius / (d + radius); }
        default: { return 1.0; }
    }
}

// Bounded uniform-grid cell coordinates for a point, clamped to the domain.
fn cell_coords(p: vec3<f32>, grid_min: vec3<f32>, cell_size: f32, dims: vec3<u32>) -> vec3<i32> {
    let rel = (p - grid_min) / cell_size;
    return vec3<i32>(
        clamp(i32(floor(rel.x)), 0, i32(dims.x) - 1),
        clamp(i32(floor(rel.y)), 0, i32(dims.y) - 1),
        clamp(i32(floor(rel.z)), 0, i32(dims.z) - 1),
    );
}

// Flatten cell coordinates to a linear cell index (x-major).
fn cell_index(c: vec3<u32>, dims: vec3<u32>) -> u32 {
    return c.x + c.y * dims.x + c.z * dims.x * dims.y;
}

// Linear cell index for a point (clamp + flatten in one step).
fn cell_of(p: vec3<f32>, grid_min: vec3<f32>, cell_size: f32, dims: vec3<u32>) -> u32 {
    return cell_index(vec3<u32>(cell_coords(p, grid_min, cell_size, dims)), dims);
}
