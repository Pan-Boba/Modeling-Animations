import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initial
x0, p0 = 0, 1
m, l , g = 1, 9.8, 9.8
time, n = 50, 2001

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

sol = solve(x0, p0)
x, v = sol[:, 0], sol[:, 1] / m

fig = plt.figure(figsize=(7, 7))

def update(num, x):
    global l
    plt.clf()
    plt.plot([0, -l * np.sin(x[num])], [0, -l * np.cos(x[num])], 'k')
    plt.plot(-l * np.sin(x[num]), -l * np.cos (x[num]),'ro', ms=10 )
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)

anim = animation.FuncAnimation(fig, update,  fargs=[x], frames=500000, interval=1, repeat=True)

plt.show()
