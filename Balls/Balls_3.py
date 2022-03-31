import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib import pylab


# initial
n, M, T, V = 100, 1, 1, 1
dist = np.arange(0, 1.5, 0.03)

class ParticleBox:
    """
    init_state is an [N x 4] array, where N is the number of particles:
       [[x1, y1, vx1, vy1],
        ...            ]
    bounds is the size of the box: [xmin, xmax, ymin, ymax]
    """
    def __init__(self, init_state,
                 bounds = [-2, 2, -2, 2],
                 size = 0.05,
                 M = 0.05,
                 G = 9.8):
        self.init_state = np.asarray(init_state, dtype=float)
        self.M = M * np.ones(self.init_state.shape[0])
        self.size = size
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.bounds = bounds
        self.G = G

    def periodic_move(self, dt):
        global T, V
        self.time_elapsed += dt
        if (self.time_elapsed // T) % 2 == 0:
            self.bounds[0] += V * dt
            self.bounds[1] += V * dt
        else:
            self.bounds[0] -= V * dt
            self.bounds[1] -= V * dt

    def step(self, dt):
        # update positions
        global T, V
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

        self.state[crossed_y1 | crossed_y2, 3] *= -1
        self.state[crossed_x1 | crossed_x2, 2] *= -1

        if (self.time_elapsed // T) % 2 == 0:
            self.state[crossed_x1 | crossed_x2, 2] += V * dt
        else:
            self.state[crossed_x1 | crossed_x2, 2] -= V * dt




        # add gravity
        # self.state[:, 3] -= self.M * self.G * dt


def hist(v_x, v_y):
    global M, dist
    E, dist_E = [], []
    for i in range(len(dist)):
            dist_E.append(0)
            for j in range(n):
                E.append(M * (pow(v_x[j], 2) + pow(v_y[j], 2)) / 2)
                if (dist[i] <= E[j]) and (E[j] < dist[i+1]):
                    dist_E[i] += 1
    return dist_E, E


def Distribution(E):
    global n
    f_E= []
    a = 0.085
    for i in range(len(E)):
        f_E.append(3 * np.exp(- 0.7 * E[i] / a) / a)
    return f_E


# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((n, 4))
init_state[:, :2] *= 3.9
box = ParticleBox(init_state)
dt = 1. / 30 # 30fps


# set up figure and animation
fig = plt.figure(figsize=(5, 10))
ax1, ax2, ax3 = fig.add_subplot(311, xlim=(-3.05 + 0.5, 3.05 + 0.5), ylim=(-2.1, 2.1)),\
           fig.add_subplot(312, xlim=(0, 0.35), ylim=(0, 50)), \
           fig.add_subplot(313, xlim=(0, 1500), ylim=(0, 0.2))
ax1.axis('off')
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax1.add_patch(rect)
data = Distribution(dist)
line_1, = ax2.plot(dist, data, color='black')


# declaring objects for animation
particles, = ax1.plot([], [], 'bo', ms=5)
Bar_E = ax2.bar(dist + 0.03/2, hist(box.state[:, 2], box.state[:, 3])[0], width=0.03)
line_2, = ax3.plot([], [], color='grey', linewidth=4)


# animation

E_av, c = [], []
def animate(i):
    box.periodic_move(dt)
    rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
    ax1.add_patch(rect)

    box.step(dt)
    rect.set_edgecolor('k')
    particles.set_data(box.state[:, 0], box.state[:, 1])
    f_data = np.array(hist(box.state[:, 2], box.state[:, 3]))
    c.append(i)
    E_av.append(float(sum(f_data[1]))/ len(f_data[1]))
    line_2.set_data(c, E_av)

    for Bari_E, i in zip(Bar_E, range(len(dist))):
        Bari_E.set_height(f_data[0][i])
    return [particles] + [rect] + \
           [Bari_E for Bari_E in Bar_E] + [line_1] + [line_2]


ani = animation.FuncAnimation(fig, animate, frames=6000, interval=10, blit=True)
plt.show()
