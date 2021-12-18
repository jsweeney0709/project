# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 13:40:41 2021

@author: jswee
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Kerr

consts = [0.956545,0.95,13.4126,-0.830327]
initial = [6.5,np.pi/2,0,0]

E,a,Q,l = consts
r,th,ph,ti = initial

#for initial conditions on rdot and thetadot choose between returning +- for each

Sigma=(r**2+(a*np.cos(th))**2)
Zeta = 2*r/Sigma

#set up equations
def mydiff(lamb, X, P):
    [E,a,Q,L] = P
    [r,th,ph,ti] = X

    Sigmaover = 1/(r**2+(a*np.cos(th))**2)
    Delta = r**2-2*r+a**2
    Theta=Q-(np.cos(th)**2)*((L/np.sin(th))**2-(a*E)**2)
    Pr=E*(r**2+a**2)-a*L
    Rr=Pr**2-Delta*(Q+(L-a*E)**2)
    if Theta<0:
        Theta=-Theta
    f1=-Sigmaover*(Rr**(1/2))
    f2=-Sigmaover*(Theta**(1/2))
    f3=Sigmaover*(-a*E-L/(np.sin(th)**2)+(a*Pr)/Delta)
    f4=-a*(a*E*(np.sin(th))**2-L)+(Pr/Delta)*(r**2+a**2)
    f=[f1,f2,f3,f4]
    return f

# ODE solver parameters
stop = 400

#Define Isco
Z1 = 1+((1-a**2)**(1/3))*((1+a)**(1/3)+(1-a)**(1/3))
Z2 = (3*a**2+Z1**2)**(1/2)
Iscop = 3+Z2+((3-Z1)*(3+Z1+2*Z2))**(1/2)
Iscom = 3+Z2-((3-Z1)*(3+Z1+2*Z2))**(1/2)

def event(t,X,P):
    return 2.3 - abs(X[0])
event.terminal=True

sol = solve_ivp(mydiff, [0,stop], initial, args=(consts,), events=[event])

r=sol.y[0]
theta=sol.y[1]
phi=sol.y[2]


fig = plt.figure()
ax = plt.gca(projection='3d') 
ax.plot(r*np.cos(phi)*np.sin(theta),r*np.cos(theta)*np.sin(phi),r*np.cos(theta))
ax.plot([-6.5],[0],[0])
ax.plot([0],[-6.5],[0])
ax.plot([0],[6.5],[0])
ax.plot([0],[0],[-6.5])
ax.plot([0],[0],[6.5])

u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = (1+(1-consts[1]**2)**(1/2))*np.cos(u)*np.sin(v)
y = (1+(1-consts[1]**2)**(1/2))*np.sin(u)*np.sin(v)
z = (1+(1-consts[1]**2)**(1/2))*np.cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=1)
ax.azim = 120
ax.elev = 30
plt.show()