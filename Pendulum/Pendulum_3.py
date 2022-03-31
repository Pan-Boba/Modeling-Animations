import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

time, n = 20, 5001
m, l , g = 1, 9.8, 9.8

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

Ampl = np.arange(0, np.pi, 0.01)
t, T = [], []
delta = []

for j in range(len(Ampl)):
    sol = solve(Ampl[j], 0)
    for i in range(len(sol[:, 1])-3):
        if i != 0 and abs((sol[:,1][i+1]-sol[:,1][i]) <= abs((sol[:,1][i+2]-sol[:,1][i+1])))\
                and abs((sol[:,1][i+2]-sol[:,1][i+1]) >= abs((sol[:,1][i+3]-sol[:,1][i+2]))):
            T.append(2*(i+1))
    stop = int(min(T) / 2)
    v_av = sum(((sol[:, 1][j]) for j in range(0, stop))) / (stop * m)
    L = 2  * sol[:, 0][stop]
    T_half = L / v_av
    t.append(2* T_half)
    T.clear()


plt.plot(Ampl, t, 'b')
plt.xlabel('x')
plt.ylabel('t')
plt.grid()
plt.axis([0, np.pi, 0, 10])

plt.show()
