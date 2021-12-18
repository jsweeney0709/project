# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 17:23:47 2021

@author: jswee
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

initial=[6.56906,0,0,-0.5,(np.pi/2)-(11/50),0]
[x,y,z,v,delta,alpha] = initial
a = 0.95

r=((x**2+y**2+z**2-a**2+((x**2+y**2+z**2-a**2)**2+4*(a*z)**2)**(1/2))/2)**(1/2)
if x>0:
    phi=np.tan(y/x)
elif x<0:
    phi=np.tan(y/x)+np.pi
else:
    phi=np.pi/2
theta=np.arccos(z/r)


Sigma = (a*np.cos(theta))**2+r**2
Delta = a**2+r**2-2*r
Chi = (a**2+r**2)**2-Delta*(a*np.sin(theta))**2
Zeta = 2*r/Sigma

L = v*np.cos(alpha)*np.cos(delta)*np.sin(theta)*(Chi/Sigma)**(1/2)
E = ((Delta*Sigma*Chi)**(1/2))/(Chi+2*r*a*L)
Q = (L*np.tan(delta))**2-(E*a*np.sin(delta))**2
k = (a*E)**2+L**2+Q

Ptheta = v*np.cos(alpha)*np.sin(delta)*Sigma**(1/2)
PthetaDot = (np.sin(theta)*np.cos(theta)/Sigma)*((L/(np.sin(theta)**2))**2-(a*E)**2)
Pphi = L
PphiDot = 0
Pr = v*np.sin(alpha)*(Sigma/Delta)**(1/2)
PrDot = (2*(E**2)*r*(a**2+r**2)-k*(r-1)-2*a*E*L)/(Sigma*Delta)-2*(r-1)*(Pr**2)/Sigma
varlist = [r, theta, phi, Pr, Ptheta]
constlist = [E,a,L,k]
print(L,E,Q)


def mydiff(t, X=varlist, P=constlist):
    [E,a,L,k] = P
    rt, thetat, phit, Prt, Pthetat = X
    Sigmat = (a*np.cos(thetat))**2+r**2
    Deltat = a**2+rt**2-2*rt
    PthetaDot = (np.sin(thetat)*np.cos(thetat)/Sigmat)*((L/(np.sin(thetat)**2))**2-(a*E)**2)
    PrDot = (2*(E**2)*rt*(a**2+rt**2)-k*(rt-1)-2*a*E*L)/(Sigmat*Deltat)-2*(rt-1)*(Prt**2)/Sigmat
    rDot = Delta*Prt/Sigma
    thetaDot = Pthetat/Sigma
    phiDot = (2*a*E*rt+L*(np.sin(thetat)**(-2))*(Sigma-2*rt))/(Delta*Sigma)
    f = [rDot,thetaDot,phiDot,PrDot, PthetaDot]
    return f

# ODE solver parameters
stop =100
numpoints = 400

# Form time basis
t = np.linspace(0,stop,numpoints)

#Define Isco
Z1 = 1+((1-a**2)**(1/3))*((1+a)**(1/3)+(1-a)**(1/3))
Z2 = (3*a**2+Z1**2)**(1/2)
Iscop = 3+Z2+((3-Z1)*(3+Z1+2*Z2))**(1/2)
Iscom = 3+Z2-((3-Z1)*(3+Z1+2*Z2))**(1/2)

def event(t,X,P):
    return abs(X[0])-abs(1+(1-P[1]**2)**(1/2))
event.terminal=True

sol = solve_ivp(mydiff, [0,stop], varlist, args=(constlist,), events=[event])

r=sol.y[0]
theta=sol.y[1]
phi=sol.y[2]


fig = plt.figure()
ax = plt.gca(projection='3d') 
ax.plot(r*np.cos(phi)*np.sin(theta),r*np.cos(theta)*np.sin(phi),r*np.cos(theta))
ax.plot([-40],[0],[0])
ax.plot([40],[0],[0])
ax.plot([0],[-40],[0])
ax.plot([0],[40],[0])
ax.plot([0],[0],[-40])
ax.plot([0],[0],[40])

u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
x = (1+(1-a**2)**(1/2))*np.cos(u)*np.sin(v)
y = (1+(1-a**2)**(1/2))*np.sin(u)*np.sin(v)
z = (1+(1-a**2)**(1/2))*np.cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=0.8)
ax.azim = 90
ax.elev = 10
plt.show()

