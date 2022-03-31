import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initial
x0, p0 = 0, 0
m, k = 1, 0.01
time, n = 100, 10001

def eq(y, t):
    global m, k
    x, p = y
    f = 2 * np.cos(x) * np.sin(5 * t)
    dydt = [p/m, - np.sin(x) - k * p + f]
    return dydt

def solve(x0, p0):
    global time, n, m
    cond = [x0, p0/m]
    t = np.linspace(0, time, n)
    sol = odeint(eq, cond, t)
    return sol

# figure
fig = plt.figure()

plt.title('Trajectory', fontsize = 14)
plt.grid()
plt.xlabel('t')
plt.ylabel('x')


# variables
sol = solve(x0, p0)
t = np.linspace(0, time, n)
x, p = sol[:, 0], sol[:, 1]

plt.plot(t, x, color='k')
# declaring lines for animation
#line, = plt.plot(t, x, color='k')
#mk, = plt.plot([], [], 'ro', ms =10)

#def update(num, x, t, line):
    #plt.clf()
#    line.set_data(t[:num], x[:num])
#    mk.set_data(t[num-1], x[num-1])
#    #line.axes.axis([-2, 2, -2, 2])
#    return line, mk,

#animation.FuncAnimation(fig, update, len(x), fargs=[x, t, line], interval=1, blit=True)

plt.show()
