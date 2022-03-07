# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:29:37 2021

@author: jswee
"""
import matplotlib.pyplot as plt
import numpy as np



X0 = [10, -1, np.pi]
P = [1, -8]
M, L = P

stoptime = 20
numpoints = 1500

# Form time basis
t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

def solver(X):
    Rlst=[X[0]]
    Slst=[X[1]]
    Plst=[X[2]]
    for time in range(len(t)-1):
        Rlst.append(Rlst[time]+Slst[time]*(t[time+1]-t[time]))
        Slst.append(Slst[time]-(L**2*(3*M-Rlst[time])/(Rlst[time]**4)*(t[time+1]-t[time])))
        Plst.append(Plst[time]+(L/Rlst[time]**2)*(t[time+1]-t[time]))
    return [Rlst,Slst,Plst]


sol = solver(X0)

r = sol[0]
phi = sol[2]

# Stop light when it reaches the event horizon
for i in range(len(r)):
    q=0
    if r[i] <= 2*P[0]:
        a=r[:i]
        b=phi[:i]
        q+=1
        break
if q == 0:
    a = r
    b = phi
    
# Set up plot
figure, ax = plt.subplots()
ax.axis("equal")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.plot(a*np.cos(b),a*np.sin(b))
plt.plot(a[-1]*np.cos(b[-1]),a[-1]*np.sin(b[-1]), 'rp')
disk1=plt.Circle((0, 0), 2*P[0], color="k", fill=True)
ax.add_artist(disk1)
    