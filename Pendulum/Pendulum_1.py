import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initial
x0, p0 = 0, 1
m, l , g = 1, 9.8, 9.8
time, n = 100, 1001

def eq(y, t):
    global m
    x, p = y
    dydt = [p/m, - np.sin(x)]
    return dydt

def solve(x0, p0):
    global time, n, m
    cond = [x0, p0/m]
    t = np.linspace(0, time, n)
    sol = odeint(eq, cond, t)
    return sol

# figure
gridsize = (2, 1)
fig = plt.figure(figsize=(10, 10))
ax1, ax2 = plt.subplot2grid(gridsize, (0, 0)), plt.subplot2grid(gridsize, (1, 0))

ax1.set_title('Trajectory in phase space', fontsize = 14)
ax1.grid()
ax1.set_xlabel('x')
ax1.set_ylabel('p')

ax2.set_title('Energy conservation', fontsize = 14)
ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('E')

# variables
sol = solve(1, 1)
x, p = sol[:, 0], sol[:, 1]
E = []

for i in range(len(x)):
    E.append(pow(p[i], 2) * l * l / (2 * m) + m * l * g * ( 1 - np.cos(x[i])))

# declaring lines for animation
line, = ax1.plot(x, p, color='k')
mk, = ax1.plot([], [], 'ro', ms =10)
line_1, = ax2.plot(x, E, color='k')
mk_1, = ax2.plot([], [], 'ro', ms =10)

def update(num, x, p, E, line, line_1):
    line.set_data(x[:num], p[:num])
    mk.set_data(x[num-1], p[num-1])
    line.axes.axis([-2, 2, -2, 2])
    line_1.set_data(x[:num], E[:num])
    mk_1.set_data(x[num-1], E[num-1])
    line.axes.axis([-2, 2, -2, 2])
    return line, mk, line_1, mk_1,

animation.FuncAnimation(fig, update, len(x), fargs=[x, p, E, line, line_1], interval=20, blit=True)

plt.subplots_adjust(wspace=0, hspace=0.6)
plt.show()
