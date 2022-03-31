import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#initial
n, dh = 500, 0.01
borders, t = np.arange(-1.5, 1.5, 0.1), np.arange(0, 200, 0.1)
x_av, x2_av = [], []


class ParticleBox:
    def __init__(self, init_state):
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()

    def step(self):
        global dh
        # update positions
        for i in range(len(init_state)):
            self.state[i, 0] += dh * 2 * np.random.uniform(-1, 1)

    def average(self):
        global n
        x_av = sum(self.state[0:(n-1), :1])
        #x2_av = sum()
        return x_av

    def sqr_average(self):
        global n
        x2_av = sum(pow(self.state[0:(n-1), :1], 2))
        return x2_av


# set up initial state
init_state = []
for i in range(n):
    init_state.append([0, np.random.uniform(0, 500)])
box = ParticleBox(init_state)


# figure and grey lines
fig = plt.figure(figsize=(10, 10))
ax1, ax2, ax3 = fig.add_subplot(311, xlim=(-1.5, 1.5), ylim=(0, 500)), \
                fig.add_subplot(312, xlim=(0, 200), ylim=(-15, 15)),  \
                fig.add_subplot(313, xlim=(0, 200), ylim=(0, 150))
for i in range(len(borders)):
    ax1.vlines(borders[i], -5, 505, color='lightgrey')
ax2.hlines(0, -1, 205, color='lightgrey')
ax3.plot([0, 300], [0, 200], color='lightgrey')


# declaring objects for animation
particles, = ax1.plot([], [], 'bo')
mk, = ax2.plot([], [], 'ro', ms =10)
line, = ax2.plot(t, [0] * len(t), color='k')
mk_1, = ax3.plot([], [], 'ro', ms =10)
line_1, = ax3.plot(t, [0] * len(t), color='k')


# animation
def animate(i):
    box.step()
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(5)

    x_av.append(box.average())
    line.set_data(t[:i], x_av[:i])
    mk.set_data(t[i], x_av[i])

    x2_av.append(box.sqr_average())
    line_1.set_data(t[:i], x2_av[:i])
    mk_1.set_data(t[i], x2_av[i])

    return particles, line, mk, line_1, mk_1


ani = animation.FuncAnimation(fig, animate, frames=60000, interval=10, blit=True)
plt.show()
