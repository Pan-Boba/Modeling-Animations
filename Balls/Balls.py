import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib import pylab


# initial
n = 100
radius = np.arange(0, 2, 0.2)


class ParticleBox:
    """
    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        ...            ]
    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self, init_state,
                 bounds = [-2, 2, -2, 2],
                 size = 0.04,
                 M = 0.05,
                 G = 9.8):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def step(self, dt):
        self.time_elapsed += dt

        # update positions
        self.state[:, :2] += dt * self.state[:, 2:]

        # find pairs of particles undergoing a collision
        D = squareform(pdist(self.state[:, :2]))
        ind1, ind2 = np.where(D < 2 * self.size)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        for i1, i2 in zip(ind1, ind2):
            # mass
            m1 = self.M[i1]
            m2 = self.M[i2]

            # location vector
            r1 = self.state[i1, :2]
            r2 = self.state[i2, :2]

            # velocity vector
            v1 = self.state[i1, 2:]
            v2 = self.state[i2, 2:]

            # relative location & velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            v_rel = 2 * r_rel * vr_rel / rr_rel - v_rel

            # assign new velocities
            self.state[i1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[i2, 2:] = v_cm - v_rel * m1 / (m1 + m2)

        # check for crossing boundary
        crossed_x1 = (self.state[:, 0] < self.bounds[0] + self.size)
        crossed_x2 = (self.state[:, 0] > self.bounds[1] - self.size)
        crossed_y1 = (self.state[:, 1] < self.bounds[2] + self.size)
        crossed_y2 = (self.state[:, 1] > self.bounds[3] - self.size)

        self.state[crossed_x1, 0] = self.bounds[0] + self.size
        self.state[crossed_x2, 0] = self.bounds[1] - self.size

        self.state[crossed_y1, 1] = self.bounds[2] + self.size
        self.state[crossed_y2, 1] = self.bounds[3] - self.size

        self.state[crossed_x1 | crossed_x2, 2] *= -1
        self.state[crossed_y1 | crossed_y2, 3] *= -1



        # add gravity
        # self.state[:, 3] -= self.M * self.G * dt


# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((n, 4))
init_state[:, :2] *= 3.9
box = ParticleBox(init_state, size=0.04)
dt = 1. / 30 # 30fps


# set up figure
fig = plt.figure(figsize=(5, 10))
ax1, ax2 = fig.add_subplot(211, xlim=(-2.05, 2.05), ylim=(-2.05, 2.05)),\
           fig.add_subplot(212, xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
ax1.axis('off')
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax1.add_patch(rect)
for i in range(len(radius)):
    Circle = plt.Circle((0, 0), radius=radius[i], fill=False, color='lightgrey')
    ax2.add_artist(Circle)


# declaring objects for animation
particles, = ax1.plot([], [], 'bo', ms=6)
vel_distribution, = ax2.plot([], [], 'bo', color='black', ms=3)


# animation
def animate(i):
    box.step(dt)
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(8)

    vel_distribution.set_data(box.state[:, 2], box.state[:, 3])

    return particles, rect, vel_distribution


ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, blit=True)
plt.show()
