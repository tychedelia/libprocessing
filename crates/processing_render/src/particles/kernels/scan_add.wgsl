// Adds each block's already-scanned exclusive offset (`block_sums[workgroup]`)
// to every element of that block. Second half of the multi-level scan
// primitive (see `particles/scan.rs`); run once per level, top-down.

@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read> block_sums: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = arrayLength(&data);
    let g = gid.x;
    if g >= n { return; }
    data[g] = data[g] + block_sums[wid.x];
}
