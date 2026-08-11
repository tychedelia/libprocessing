// Stream compaction, step 3: each kept particle writes its own index into its
// dense output slot — its exclusive-scan offset. After this,
// `indices[0..count]` are the kept particle indices in ascending order.
// See `particles/compact.rs`.

@group(0) @binding(0) var<storage, read>       flags:   array<f32>;
@group(0) @binding(1) var<storage, read>       scanned: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&flags) { return; }
    if flags[i] != 0.0 {
        indices[scanned[i]] = i;
    }
}
