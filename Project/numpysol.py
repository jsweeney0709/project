# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 14:37:45 2021

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
stoptime = 1000
numpoints = 2000

# Form time basis
t = np.linspace(0,stoptime,numpoints)

# Initial conditions
X0 = [100, -1, np.pi]
alpha=0.21
L=(X0[0]*X0[1]*np.cos(X0[2])*np.tan(alpha)-X0[0]*X0[1]*np.sin(X0[2]))/(np.cos(X0[2])+np.sin(X0[2])*np.tan(alpha))
P = [4, L]


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

k=0
for j in range(1,len(b)):
    k+=b[j]-b[j-1]
    
#useful function
def findAngle(x0,y0,x1,y1):
    p=x1-x0
    q=y1-y0
    tan=np.arctan(abs(q)/abs(p))
    if p<0:
        if q<0:
            answer = tan-np.pi
        elif q>=0:
            answer = np.pi-tan
    elif p>=0:
        if q<0:
            answer = -tan
        elif q>=0:
            answer = tan
    return answer+np.ceil(k/(2*np.pi))*(2*np.pi)
    
# Set up plot
figure, ax = plt.subplots()
ax.axis("equal")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Function to help clean code up
def findx(i):
    return a[i]*np.cos(b[i])
def findy(i):
    return a[i]*np.sin(b[i])


# Find vector from origin
def g(x):
    x1=findx(0)
    y1=findy(0)
    x2=findx(-1)
    y2=findy(-1)
    m= (y2-y1)/(x2-x1)
    return m*x+y1-m*x1

#xlist=[findx(0)+((findx(-1)-findx(0))*i/numpoints) for i in range(numpoints)]
xlist=np.linspace(a[0]*np.cos(b[0]),a[-1]*np.cos(b[-1]))
yval=[a[0]*np.sin(b[0])-np.tan(alpha)*a[0]*np.cos(b[0])+x*np.tan(alpha) for x in xlist]
# Plot
plt.plot(a*np.cos(b),a*np.sin(b))
plt.plot(findx(0),findy(0), 'rp')
plt.plot(xlist,yval, linestyle='dotted')
plt.plot(xlist,[g(x) for x in xlist], linestyle='dotted')
disk1=plt.Circle((0, 0), 2*P[0], color="k", fill=True)
ax.add_artist(disk1)

# Show
plt.show()
#print(findAngle(a[0]*np.cos(b[0]),a[0]*np.sin(b[0]),xlist[-1],yval[-1]),findAngle(a[0]*np.cos(b[0]),a[0]*np.sin(b[0]),a[-1]*np.cos(b[-1]),a[-1]*np.sin(b[-1])))