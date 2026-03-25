from mewnala import *
import random

particles = []
origin_x = 0
origin_y = 50


def setup():
    global origin_x
    size(640, 360)
    origin_x = 640 / 2


def draw():
    background(0)
    particles.append(Particle(origin_x, origin_y))
    for p in particles:
        p.update()
        p.display()
    for i in range(len(particles) - 1, -1, -1):
        if particles[i].is_dead():
            particles.pop(i)


class Particle:
    def __init__(self, x, y):
        self.px = x
        self.py = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-2, 0)
        self.ax = 0
        self.ay = 0.05
        self.lifespan = 255.0

    def update(self):
        self.vx += self.ax
        self.vy += self.ay
        self.px += self.vx
        self.py += self.vy
        self.lifespan -= 1.0

    def display(self):
        stroke(255, self.lifespan)
        fill(255, self.lifespan)
        ellipse(self.px, self.py, 8, 8)

    def is_dead(self):
        return self.lifespan < 0.0


run()
