# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 15:12:39 2022

@author: jswee
"""

from numpy import pi, sin, cos, sqrt, mgrid
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

r,theta,phi = [70,pi/2,0]
v=1
a=0.998
incl=pi
#run from 170-370
alpha=265*pi/180
x=sqrt(r**2+a**2)*sin(theta)*cos(phi)
y=sqrt(r**2+a**2)*sin(theta)*sin(phi)
z=r*cos(theta)
vr=v*sin(alpha)
vp=v*cos(alpha)*sin(incl)
vt=v*cos(alpha)*cos(incl)
def Sigma(r,theta):
    return (a*cos(theta))**2+r**2
def Delta(r):
    return a**2+r**2-2*r
def Chi(r,theta):
    return (a**2+r**2)**2-Delta*(a*sin(theta))**2
def Zeta(r):
    return 2*r/Sigma
bl=sqrt(((r**2+a**2)**2-a**2*Delta(r)*sin(theta)**2)/(r**2+a**2*cos(theta)**2))*sin(theta)
L=vp*bl
drag=2*r*a/((r**2+a**2)**2-a**2*(r**2+a**2-2*r)*sin(theta)**2)
epsilon=sqrt(Delta(r)*Sigma(r,theta)/((a**2+r**2)**2-a**2*Delta(r)*sin(theta)**2))+L*drag
ptheta=vt*sqrt((x**2+y**2+z**2))
pr=vr*sqrt((Sigma(r,theta)/Delta(r)))
Q=ptheta**2+((L/sin(theta))**2-a**2*epsilon**2*cos(theta)**2)
k=Q+L**2+a**2*epsilon**2




# ODE solver parameters
stop = 150

varlist = [0,r,theta,phi,pr,ptheta]

def event(tau,X):
    return abs(X[1])-abs(1+(1-(a*cos(X[2]))**2)**(1/2))
event.terminal=True

def mydiff(tau, X=varlist):
    t,r,theta,phi,pr,ptheta=X
    td=epsilon+(2*r*(r**2+a**2)*epsilon-2*a*r*L)/(Sigma(r,theta)*Delta(r))
    rd=pr*Delta(r)/Sigma(r,theta)
    thetad=ptheta/Sigma(r,theta)
    phid=(2*a*r*epsilon+(Sigma(r,theta)-2*r)*L/(sin(theta)**2))/(Sigma(r,theta)*Delta(r))
    prd=1/(Sigma(r,theta)*Delta(r))*(-k*(r-a)+2*r*(r**2+a**2)*epsilon**2-2*a*epsilon*L)-(2*pr**2*(r-1))/Sigma(r,theta)
    pthetad=(sin(theta)*cos(theta))/Sigma(r,theta)*(L**2/(sin(theta)**4)-(a*epsilon)**2)
    return [td,rd,thetad,phid,prd,pthetad]


sol = solve_ivp(mydiff, [0,stop], varlist, events=[event], method='LSODA')

r=sol.y[1]
theta=sol.y[2]
phi=sol.y[3]


fig = plt.figure()
ax = plt.gca(projection='3d') 
ax.plot(sqrt(r**2+a**2)*cos(phi)*sin(theta),sqrt(r**2+a**2)*cos(theta)*sin(phi),r*cos(theta))
ax.plot([-10],[0],[0])
ax.plot([10],[0],[0])
ax.plot([0],[-10],[0])
ax.plot([0],[10],[0])
ax.plot([0],[0],[-10])
ax.plot([0],[0],[10])

u, v = mgrid[0:2*pi:50j, 0:pi:50j]
x = (1+(1-(a*cos(v))**2)**(1/2))*cos(u)*sin(v)
y = (1+(1-(a*cos(v))**2)**(1/2))*sin(u)*sin(v)
z = (1+(1-(a*cos(v))**2)**(1/2))*cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=0.8)
ax.azim = 90
ax.elev = 10
plt.show()


