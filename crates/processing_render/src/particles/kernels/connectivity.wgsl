// Generates a hardware index buffer + indirect draw args for the direct-
// rasterization path (task #31, rung 2), so a particle position buffer can be
// drawn as connected lines/triangles with one `draw_indexed_indirect` — the
// connectivity and the draw count both come from the GPU, never the CPU.
//
// `indices` is bound as storage here but also carries `INDEX` usage; `args` also
// carries `INDIRECT` usage and holds a `DrawIndexedIndirectArgs` (5x u32:
// index_count, instance_count, first_index, base_vertex, first_instance). See
// `particles/connectivity.rs`. Vertices are pulled by index in `point.wgsl`.

struct Params {
    count: u32,  // number of particles (vertices)
    mode: u32,   // 0 = line chain (i -> i+1)
}

const MODE_LINE_CHAIN: u32 = 0u;

@group(0) @binding(0) var<storage, read_write> indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> args:    array<u32>;
@group(0) @binding(2) var<uniform>             params:  Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.count;
    if n == 0u { return; }

    // A polyline through consecutive particles: segment i joins vertex i to i+1,
    // emitting a LINE_LIST pair. n particles -> n-1 segments -> 2*(n-1) indices.
    if params.mode == MODE_LINE_CHAIN {
        if i + 1u < n {
            indices[i * 2u]      = i;
            indices[i * 2u + 1u] = i + 1u;
        }
    }

    // One thread writes the indirect draw args. index_count matches the emitted
    // connectivity; the draw is a single non-instanced pass.
    if i == 0u {
        var index_count = 0u;
        if params.mode == MODE_LINE_CHAIN {
            index_count = select(0u, (n - 1u) * 2u, n > 1u);
        }
        args[0] = index_count;  // index_count
        args[1] = 1u;           // instance_count
        args[2] = 0u;           // first_index
        args[3] = 0u;           // base_vertex
        args[4] = 0u;           // first_instance
    }
}
