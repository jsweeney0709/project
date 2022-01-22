# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 15:12:39 2022

@author: jswee
"""

from numpy import pi, sin, cos, sqrt, mgrid, arctan, arccos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

r, theta, phi = [100, pi/2, pi/2]
v = 1.001
a = 1

#-46.83744492589279 -3.098841465049254 -8.74478540492632
#1059 2
#0.10737865515199488 1.564660403643354
#-0.02106273580687733 3.201074567558693

x=272
y=0
height=1024

alpha = sqrt(y**2+x**2)*pi/height
if x>0:
    beta=arctan(y/x)
elif x<0:
    beta=pi+arctan(y/x)
if x==0:
    if y>0:
        beta=pi/2
    else:
        beta=3*pi/2
print(arccos(-1))

alpha = pi
beta = 0

launch = 3*pi/2 - alpha
if alpha<0:
    incl = -beta - pi/2
else:
    incl = beta - pi/2
    

x = sqrt(r**2+a**2)*sin(theta)*cos(phi)
y = sqrt(r**2+a**2)*sin(theta)*sin(phi)
z = r*cos(theta)
vr = v*sin(launch)
vphi = v*cos(launch)*sin(incl)
vtheta = v*cos(launch)*cos(incl)
def Sigma(r, theta):
    return (a*cos(theta))**2 + r**2
def Delta(r):
    return a**2 + r**2 - 2*r
def Chi(r, theta):
    return (r**2 + a**2)**2 - Delta(r)*(a*sin(theta))**2
def omega(r, theta):
    return 2*r*a/Chi(r, theta)

L = vphi*sin(theta)*sqrt(Chi(r, theta)/Sigma(r, theta))
E = sqrt(Delta(r)*Sigma(r, theta)/Chi(r, theta)) + L*omega(r, theta)
ptheta = vtheta*sqrt(Sigma(r, theta))
pr = vr*sqrt((Sigma(r, theta)/Delta(r)))
Q = ptheta**2 + ((L/sin(theta))**2 - (a*E)**2)*(cos(theta)**2)
k = Q + L**2 + a**2*E**2
varlist = [0, r, theta, phi, pr, ptheta]

# ODE solver parameters
stop = 200

def event(tau, X):
    return abs(X[1]) - abs(1 + sqrt(1 - (a*cos(X[2]))**2))
event.terminal = True

def mydiff(tau, X=varlist):
    t, r, theta, phi, pr, ptheta = X
    td = E + (2*r*(r**2 + a**2)*E - 2*a*r*L)/(Sigma(r, theta)*Delta(r))
    rd = pr*Delta(r)/Sigma(r, theta)
    thetad = ptheta/Sigma(r, theta)
    phid = (2*a*r*E + (Sigma(r, theta) - 2*r)*L/(sin(theta)**2))/(Sigma(r, theta)*Delta(r))
    prd = 1/(Sigma(r, theta)*Delta(r))*(-k*(r - 1)+2*r*(r**2 + a**2)*E**2 - 2*a*E*L) - (2*pr**2*(r - 1))/Sigma(r, theta)
    pthetad = (sin(theta)*cos(theta))/(Sigma(r, theta))*(L**2/(sin(theta)**4) - (a*E)**2)
    return [td, rd, thetad, phid, prd, pthetad]


sol = solve_ivp(mydiff, [0,stop], varlist, events=[event])


r = sol.y[1]
theta=sol.y[2]
phi=sol.y[3]

x1 = sqrt(r[-1]**2+a**2)*sin(theta[-1])*cos(phi[-1])
y1 = sqrt(r[-1]**2+a**2)*sin(theta[-1])*sin(phi[-1])
z1 = r[-1]*cos(theta[-1])
print(x1,y1,z1)

fig = plt.figure()
ax = plt.gca(projection='3d') 
ax.plot(sqrt(r**2 + a**2)*cos(phi)*sin(theta), sqrt(r**2 + a**2)*sin(theta)*sin(phi), r*cos(theta))

ax.plot([-4],[0],[0])
ax.plot([4],[0],[0])
ax.plot([0],[-4],[0])
ax.plot([0],[4],[0])
ax.plot([0],[0],[-4])
ax.plot([0],[0],[4])

u, v = mgrid[0:2*pi:50j, 0:pi:50j]
rr = sqrt(1 + (1 - a**2))
x = sqrt(rr**2 + a**2)*cos(u)*sin(v)
y = sqrt(rr**2 + a**2)*sin(u)*sin(v)
z = rr*cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=1)

ax.azim = 0
ax.elev = 90
#ax.set_axis_off()
plt.show()


