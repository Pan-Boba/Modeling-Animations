import numpy as np
from math import sqrt
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initial
x0, p0 = 0, 0
m, k, f = 1, 0.095, 0.26
time, n = 100, 1001
w = np.arange(0.6, 1.1, 0.005)

def eq(y, t, w, q):
    global m, k, f, time
    phi = 0.005 * (q-1) * time
    x, p = y
    dydt = [p/m, - np.sin(x) - 2 * k * p + f * np.sin(w * t - phi) * np.cos(x)]
    return dydt

def solve(w, x0, p0, q):
    global time, n, m
    t = np.linspace((q-1) * time, q * time, n)
    sol = odeint(eq, [x0, p0/m], t, args = (w, q))
    return sol

def theoretical():
    global m, k, f
    a = np.arange(0.05, 1.0, 0.001)
    w1, w2, D, aa_1, aa_2 = [], [], [], [], []
    for i in range(len(a)):
        D.append(4 * pow(k, 4) - 4 * pow(k, 2) * (1 - 0.5 * pow(a[i], 2)) + pow(f, 2)/ (4 * pow(a[i], 2)))
        if D[i] >= 0:
            aa_2.append(2 * a[i])
            if (1 - 2 * pow(k, 2) - 0.5 * pow(a[i], 2) - sqrt(D[i])) >= 0:
                aa_1.append(2 * a[i])
                w1.append(sqrt(1 - 2 * pow(k, 2) - 0.5 * pow(a[i], 2) - sqrt(D[i])))
                w2.append(sqrt(1 - 2 * pow(k, 2) - 0.5 * pow(a[i], 2) + sqrt(D[i])))
            else:
                w2.append(sqrt(1 - 2 * pow(k, 2) - 0.5 * pow(a[i], 2) + sqrt(D[i])))

    plt.plot(w1, aa_1, color='b')
    plt.plot(w2, aa_2, color='r')

# figure
fig = plt.figure()

plt.grid()
plt.xlabel('a')
plt.ylabel('w')


# variables
a = []
for i in range(len(w)):
    try:
        sol = solve(w[i], x[1000], p[1000], (i+1))
    except NameError:
        sol = solve(w[i], 0, 0, (i+1))
    x, p = sol[:, 0], sol[:, 1]
    a.append(abs(max(x[900:1000])))

plt.axis([0.6, 1.1, 0, 2])
plt.plot(w, a, color='k')

theoretical()

plt.show()
