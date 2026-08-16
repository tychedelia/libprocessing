//! Renders an animated plasma pattern to a video file.
//!
//! cargo run -p processing_video --example record_demo -- out.mp4

use processing_video::{VideoRecorder, VideoRecorderConfig};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FPS: f64 = 30.0;
const SECONDS: f64 = 6.0;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "record_demo.mp4".to_string());

    let config = VideoRecorderConfig::new(WIDTH, HEIGHT, FPS);
    let mut recorder = VideoRecorder::new(&path, config).expect("create recorder");

    let frames = (SECONDS * FPS) as u64;
    let mut pixels = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    for i in 0..frames {
        let t = i as f32 / FPS as f32;
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let (fx, fy) = (x as f32, y as f32);
                let v = (fx / 32.0 + t).sin()
                    + (fy / 24.0 - t * 1.3).sin()
                    + ((fx + fy) / 48.0 + t * 0.7).sin()
                    + ((fx * fx + fy * fy).sqrt() / 24.0 - t * 2.0).sin();
                let p = &mut pixels[((y * WIDTH + x) * 4) as usize..][..4];
                p[0] = ((v * std::f32::consts::PI).sin() * 127.0 + 128.0) as u8;
                p[1] = ((v * std::f32::consts::PI + 2.0).sin() * 127.0 + 128.0) as u8;
                p[2] = ((v * std::f32::consts::PI + 4.0).sin() * 127.0 + 128.0) as u8;
                p[3] = 255;
            }
        }
        recorder.record_frame(pixels.clone()).expect("record frame");
    }

    let encoded = recorder.finish().expect("finish");
    println!("wrote {encoded} frames to {path}");
}
