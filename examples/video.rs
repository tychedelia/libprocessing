use processing_glfw::GlfwContext;

use processing::prelude::*;
use processing_render::render::command::DrawCommand;
use processing_video::{video_create, video_destroy, video_image, video_is_loaded, video_load};

fn main() {
    match sketch() {
        Ok(_) => {
            eprintln!("Sketch completed successfully");
            exit(0).unwrap();
        }
        Err(e) => {
            eprintln!("Sketch error: {:?}", e);
            exit(1).unwrap();
        }
    };
}

fn sketch() -> error::Result<()> {
    let width = 640;
    let height = 480;

    let mut glfw_ctx = GlfwContext::new(width, height)?;
    init(Config::default())?;

    let surface = glfw_ctx.create_surface(width, height)?;
    let graphics = graphics_create(surface, width, height, TextureFormat::Rgba16Float)?;

    let handle = video_load("video/file_example_MP4_640_3MG.mp4")?;
    let video = video_create(handle)?;
    let mut image_entity = None;

    while glfw_ctx.poll_events() {
        graphics_begin_draw(graphics)?;

        if image_entity.is_none() && video_is_loaded(video)? {
            image_entity = Some(video_image(video)?);
        }

        if let Some(img) = image_entity {
            graphics_record_command(graphics, DrawCommand::BackgroundImage(img))?;
        }

        graphics_end_draw(graphics)?;
    }

    video_destroy(video)?;
    Ok(())
}
