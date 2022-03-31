import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from matplotlib import pylab
from scipy.optimize import curve_fit


# initial
n = 50
amount = np.arange(0, 4, 0.05)


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


def Distribution(x, y):
    global n, amount
    r, dist, num = [], [], []
    for i in range(n):
        r.append(np.sqrt(pow(x[i], 2) + pow(y[i], 2)))
    for i in range(n):
        for j in range(n):
            if i < j:
                dist.append(r[i] - r[j])
    for i in range(len(amount)):
        num.append(0)
        for j in range(len(dist)):
            if (amount[i] <= dist[j]) and (dist[j] < amount[i+1]):
                num[i] += 0.5
    return num


def f(x, a, b, c):
    return a * np.exp(-b * x) + c


def func(x, a, b, c):
    list = []
    for i in range(len(x)):
        list.append(float(a * np.exp(-b * x[i]) + c))
    return list


# set up initial state
np.random.seed(0)
init_state = -0.5 + np.random.random((n, 4))
init_state[:, :2] *= 3.9
box = ParticleBox(init_state)
dt = 1. / 30 # 30fps


# set up figure
fig = plt.figure(figsize=(5, 10))
ax1, ax2 = fig.add_subplot(211, xlim=(-2.05, 2.05), ylim=(-2.05, 2.05)),\
           fig.add_subplot(212, xlim=(0, 3), ylim=(0, 30))
ax1.axis('off')
rect = plt.Rectangle(box.bounds[::2],
                     box.bounds[1] - box.bounds[0],
                     box.bounds[3] - box.bounds[2],
                     ec='none', lw=2, fc='none')
ax1.add_patch(rect)



# declaring objects for animation
particles, = ax1.plot([], [], 'bo', ms=8)
f_data = Distribution(box.state[:, 0], box.state[:, 1])
mk, = ax2.plot([], [], 'bo', ms=5, color='darkorange')

popt_1, pcov_1 = curve_fit(f, amount, f_data)
my_dict_1 = {'color' : 'black', 'linewidth' : 2.5, 'linestyle' : '--',}
line, = ax2.plot([], [], **my_dict_1)


# animation
def animate(i):
    box.step(dt)
    rect.set_edgecolor('k')
    q = box.state
    particles.set_data(q[:, 0], q[:, 1])

    f_data = Distribution(q[:, 0], q[:, 1])
    mk.set_data(amount, f_data)
    popt_1, pcov_1 = curve_fit(f, amount, f_data)
    line.set_data(amount, func(amount, *popt_1))
    return particles, rect, mk, line


ani = animation.FuncAnimation(fig, animate, frames=600, interval=10, blit=True)
plt.show()
