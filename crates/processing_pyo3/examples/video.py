from mewnala import *
from mewnala.math import *

angle = 0.0

def setup():
    global vid, paused
    size(600, 600)
    mode_3d()

    create_directional_light((1.0, 0.98, 0.95), 1500.0)

    vid = create_video("video/file_example_MP4_640_3MG.mp4")
    vid.loop()
    paused = False


def draw():
    global angle
    camera_position(0.0, 100.0, 300.0)
    camera_look_at(0.0, 0.0, 0.0)
    background(30)
    stroke(255)
    stroke_weight(2)

    if vid.is_loaded():
        texture(vid)
        texture_transform(Affine2.from_scale_angle_translation(
            Vec2(0.75, 1.0), 0.0, Vec2(0.125, 0.0)
        ))
        push_matrix()
        rotate_x(0.3)
        rotate_y(angle)
        box(100.0, 100.0, 100.0)
        pop_matrix()
        no_texture()

    angle += 0.0009


def key_pressed():
    global paused
    if key_code == SPACE:
        paused = not paused
        if paused:
            vid.pause()
        else:
            vid.play()
    elif key_code == RIGHT_ARROW:
        vid.seek(vid.position() + 1.0)
    elif key_code == LEFT_ARROW:
        vid.seek(max(0.0, vid.position() - 1.0))
    elif key_code == UP:
        vid.speed(2.0)
    elif key_code == DOWN:
        vid.speed(1.0)


run()
