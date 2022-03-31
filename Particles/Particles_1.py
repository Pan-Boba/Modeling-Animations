import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initial
n, dh = 500, 0.01
borders, dist = np.arange(-1.5, 1.5, 0.1), np.arange(-1.5, 1.5, 0.01)

class ParticleBox:
    def __init__(self, init_state):
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()

    def step(self):
        global dh, n
        v_av = []
        # update positions
        for i in range(len(init_state)):
            move = dh * 2 * np.random.uniform(-1, 1)
            self.state[i, 0] += move
            v_av.append(abs(move))
        v_av = sum(v_av) / n
        return v_av

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

def Distribution(x, t, v_av):
    global n, dh
    f = []
    q = pow((v_av * 2), 2) * t
    for i in range(len(x)):
        f.append(60 * np.exp(-pow(x[i], 2) / (2 * q)) / (np.sqrt(2 * np.pi * q)))
    return f


# set up initial state
init_state = []
for i in range(n):
    init_state.append([0, np.random.uniform(0, 500)])
box = ParticleBox(init_state)


# figure and grey lines
fig = plt.figure(figsize=(10, 10))
ax1, ax2 = fig.add_subplot(211, xlim=(-1.5, 1.5), ylim=(0, 500)),\
           fig.add_subplot(212, xlim=(-1.5, 1.5), ylim=(0, 250))
for i in range(len(borders)):
    ax1.vlines(borders[i], -5, 505, color='lightgrey')


# declaring objects for animation
f_data = np.array(box.gathering_data())[:, 0]
Bar = plt.bar(borders + 0.1, f_data, width=0.1)
particles, = ax1.plot([], [], 'bo')
line, = ax2.plot(dist, Distribution(dist, 1, box.step()), color='black')

# animation
def animate(i):
    box.step()
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(5)

    line.set_data(dist, Distribution(dist, (i+1), box.step()))

    f_data = np.array(box.gathering_data())[:, 1]
    for Bari, i in zip(Bar, range(len(borders))):
        Bari.set_height(f_data[i])
    return [particles] + [Bari for Bari in Bar] + [line]


ani = animation.FuncAnimation(fig, animate, frames=10000, interval=10, blit=True, repeat=False)
plt.show()
