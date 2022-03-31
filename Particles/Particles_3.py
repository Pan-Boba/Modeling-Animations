import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initial
n, dh = 200, 0.01
radius = np.arange(0, 2, 0.15)
dist = np.arange(0, 3, 0.01)

class ParticleBox:
    def __init__(self, init_state):
        self.init_state = np.asarray(init_state, dtype=float)
        self.state = self.init_state.copy()

    def step(self):
        global dh, n
        v_av = []
        # update positions
        for i in range(len(init_state)):
            move_x = dh * 2 * np.random.uniform(-1, 1)
            move_y = dh * 2 * np.random.uniform(-1, 1)
            move = np.sqrt(pow(move_x, 2) + pow(move_y, 2))
            self.state[i, 0] += move_x
            self.state[i, 1] += move_y
            v_av.append(move)
        v_av = sum(v_av) / n
        return v_av

    def gathering_data(self):
        global n, radius
        amount, data = [], []
        for i in range(len(radius)):
            amount.append(0)
            for j in range(n):
                position = np.sqrt(pow(self.state[j, 0], 2) + pow(self.state[j, 1], 2))
                if (radius[i] <= position) and (position < radius[i+1]):
                    amount[i] += 1
        for i in range(len(radius)):
            data.append([i, amount[i]])
        return data

def Distribution(r, t, v_av):
    global n, dh
    f = []
    q = pow((v_av * 1.1), 2) * t
    for i in range(len(r)):
        f.append(35 * r[i] * np.exp(-pow(r[i], 2) / (2 * q)) / q)
    return f


# set up initial state
init_state = []
for i in range(n):
    init_state.append([0, 0])
box = ParticleBox(init_state)


# figure and grey lines
fig = plt.figure(figsize=(5, 11))
ax1, ax2 = fig.add_subplot(211, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5)), \
           fig.add_subplot(212, xlim=(0, 2), ylim=(0, 150))
for i in range(len(radius)):
    Circle = plt.Circle((0, 0), radius=radius[i], fill=False, color='lightgrey')
    ax1.add_artist(Circle)


# declaring objects for animation
f_data = np.array(box.gathering_data())[:, 1]
Bar = plt.bar(radius  + 0.075, f_data, width=0.15)
particles, = ax1.plot([], [], 'bo')
line, = ax2.plot(dist, Distribution(dist, 1, box.step()), color='black')

# animation
def animate(i):
    box.step()
    particles.set_data(box.state[:, 0], box.state[:, 1])
    particles.set_markersize(5)

    line.set_data(dist, Distribution(dist, (i+1), box.step()))

    f_data = np.array(box.gathering_data())[:, 1]
    for Bari, i in zip(Bar, range(len(radius))):
        Bari.set_height(f_data[i])
    return [particles] + [Bari for Bari in Bar] + [line]


ani = animation.FuncAnimation(fig, animate, frames=100000, interval=25, blit=True, repeat=False)
plt.show()
