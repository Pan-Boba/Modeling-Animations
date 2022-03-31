import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

time, n, m = 20, 201, 1

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

def PhasePortrait(deltaX, deltaP, startX,  stopX, startP, stopP):
    global time, n
    for x0 in range(startX, stopX, deltaX):
            for p0 in range(startP, stopP, deltaP):
                sol = solve(x0, p0)
                plt.plot(sol[:, 0], sol[:, 1], 'b')
    plt.xlabel('x')
    plt.ylabel('p')
    plt.grid()
    plt.show()

plt.axis([-5, 5, -5, 5])

PhasePortrait(1, 1, -10, 10, -10, 10)
