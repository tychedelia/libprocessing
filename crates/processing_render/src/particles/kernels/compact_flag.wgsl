// Stream compaction, step 1: map the float 0/1 keep-flags into the u32 scan
// buffer (1 = keep), zeroing the trailing CSR slot so the exclusive scan's last
// element becomes the kept count. See `particles/compact.rs`.

@group(0) @binding(0) var<storage, read>       flags:   array<f32>;
@group(0) @binding(1) var<storage, read_write> scanned: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let m = arrayLength(&scanned); // n + 1
    if i >= m { return; }
    let n = arrayLength(&flags);
    if i < n {
        scanned[i] = select(0u, 1u, flags[i] != 0.0);
    } else {
        scanned[i] = 0u;
    }
}
