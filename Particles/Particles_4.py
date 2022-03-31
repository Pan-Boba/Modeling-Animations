import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initial
n, dh, b = 1000, 0.01, 0.001
borders, dist = np.arange(0, 8, 0.2), np.arange(0, 8, 0.01)

class ParticleBox:
    def __init__(self, init_state):
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()

    def step(self):
        global dh, n, b
        x_av, v_av = [], []
        # update positions
        for i in range(len(init_state)):
            move = dh * 2 * np.random.uniform(-1, 1) - b
            # reflection from wall
            if (self.state[i, 0] + move) < 0.06:
                move = move - (self.state[i, 0] - 0.06)
                self.state[i, 0] += move
            else: self.state[i, 0] += move
            x_av.append(self.state[i, 0])
        x_av = sum(x_av) / n
        return x_av

    def gathering_data(self):
        global n, borders
        amount, data = [], []
        for i in range(len(borders)):
            amount.append(0)
            for j in range(n):
                if (borders[i] <= self.state[j, 0]) and (self.state[j, 0] < borders[i+1]):
                    amount[i] += 1
        for i in range(len(borders)):
            data.append([i, amount[i]])
        return data

def Distribution(x, h):
    global n
    f = []
    c = n / (3 * h)
    for i in range(len(x)):
        f.append(c * np.exp(-x[i] / h))
    return f


# set up initial state
init_state = []
for i in range(n):
    init_state.append([np.random.uniform(0.06, 5.5), np.random.uniform(10, 490)])
box = ParticleBox(init_state)


# figure and grey lines
fig = plt.figure(figsize=(10, 10))
ax1, ax2 = fig.add_subplot(211, xlim=(-0.1, 6), ylim=(-10, 510)),\
           fig.add_subplot(212, xlim=(0, 6), ylim=(0, 700))
ax1.axis('off')
for i in range(len(borders)):
    ax1.vlines(borders[i], 0, 500, color='lightgrey')
ax1.hlines(500, 0, 10, color='black', linewidth = 5)
ax1.hlines(0, 0, 10, color='black', linewidth = 5)
ax1.vlines(0, -5, 505, color='black', linewidth = 5)


# declaring objects for animation
f_data = np.array(box.gathering_data())[:, 1]
Bar = plt.bar(borders + 0.1, f_data, width=0.2)
particles, = ax1.plot([], [], 'bo')
line, = ax2.plot(dist, Distribution(dist, box.step()), color='black')


# animation
def animate(i):
    box.step()
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(5)

    line.set_data(dist, Distribution(dist, box.step()))

    f_data = np.array(box.gathering_data())[:, 1]
    for Bari, i in zip(Bar, range(len(borders))):
        Bari.set_height(f_data[i])
    return [particles] + [Bari for Bari in Bar] + [line]


ani = animation.FuncAnimation(fig, animate, frames=100000, interval=10, blit=True, repeat=False)
plt.show()
