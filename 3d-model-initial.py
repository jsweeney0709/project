# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:53:03 2021

@author: jswee
"""


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


# State ODE as coupled first order
def mydiff(t, X, P):
    M, L = P
    r, s, phi = X
    f = [s, -L**2*(3*M-r)/(r**4), L/r**2]
    return f

# ODE solver parameters
stoptime = 20
numpoints = 200

# Form time basis
t = np.linspace(0,stoptime,numpoints)

# Initial conditions
X0 = [10, -1, np.pi]
alpha=np.pi/3
L=(X0[0]*X0[1]*np.cos(X0[2])*np.tan(alpha)-X0[0]*X0[1]*np.sin(X0[2]))/(np.cos(X0[2])+np.sin(X0[2])*np.tan(alpha))
P = [1, L]


#Terminate if ray hits the event horizon - for some reason doesn't work for alpha=0
def event(t,X,P):
    return abs(2*P[0])-abs(X[0])
event.terminal=True

# Solve
sol = solve_ivp(mydiff, [0,stoptime], X0, args=(P,), method='LSODA', events=[event], t_eval=t)

a = sol.y[0]
b = sol.y[2]

#Sort out when alpha=0
if alpha==0:
    sRadius=2*P[0]
    for i in range(len(a)):
        val=abs(a[i])
        if val<sRadius:
            a = a[:i]
            b = b[:i]
            break
    
# Set up plot
fig = plt.figure()
ax = plt.gca(projection='3d') 
#ax._axis3don = False

# Plot
ax.plot(a*np.cos(b),a*np.sin(b),0)
ax.plot([10],[10],[10])
ax.plot([-10],[-10],[-10])
#plt.plot(a*np.cos(b),a*np.sin(b),0)

u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=1)

#ax = fig.add_subplot(1,1,1, projection='3d')

#plot = ax.plot_surface(
#    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
#    linewidth=0, antialiased=False, alpha=0.5)
ax.azim = 90
ax.elev = 40
plt.show()