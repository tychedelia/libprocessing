from mewnala import *

def setup():
    global vid, img, paused
    size(640, 480)
    vid = create_video("video/file_example_MP4_640_3MG.mp4", looping=True)
    img = None
    paused = False


def draw():
    global img
    if img is None and vid.is_loaded():
        img = vid.image()

    if img is not None:
        background(img)


def key_pressed():
    global paused
    if key_code == SPACE:
        paused = not paused
        if paused:
            vid.pause()
        else:
            vid.resume()
    elif key_code == RIGHT_ARROW:
        vid.seek(vid.position() + 1.0)
    elif key_code == LEFT_ARROW:
        vid.seek(max(0.0, vid.position() - 1.0))
    elif key_code == UP:
        vid.set_speed(2.0)
    elif key_code == DOWN:
        vid.set_speed(1.0)


run()
