# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 17:10:46 2021

@author: jswee
"""
from numpy import sin, cos, tan, sqrt, pi, linspace, mgrid
from scipy.integrate import solve_ivp
import matplotlib as plt
plt.use('TkAgg')

varlist = [1,0,-1,5.5,-1,1,0,1]
constlist=[0.95]



def mydiff(lamb, X=varlist, P=constlist):
    xt,t,xr,r,xth,th,xph,ph = X
    a = P[0]
    rho = r**2 + a**2 * cos(th)**2
    Delta = r**2 - 2*r + a**2
    xt = sqrt(-(Delta*rho*xph**2*sin(th)**2 + 2*a**2*r*xph**2*sin(th)**2 + 2*r**3*xph**2*sin(th)**2 + rho**2*xth**2 + rho**2*xr**2/Delta)/(2*r - rho))
    
    #We have a, t,r,theta,phi
    #t'
    f1 = 2*(a**2*r*sin(2*th)*xth + (a**2*cos(th)**2 - r**2)*xr)*xt/((a**2*cos(th)**2 + r**2)*(-2*r + a**2*cos(th)**2 + r**2))
    #r'
    f2 = -(-1*r*(a**2*cos(th)**2 + r**2)**2*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2))*(-2*r + a**2 + r**2)**2*xth**2 + (16**2*a**2*r*(a**2*cos(th)**2 - r**2)*(-2*r + a**2 + r**2)**2*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(r*(-2*r + a**2 + r**2) + (- r)*(a**2*cos(th)**2 + r**2))*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2)))*xr**2 + (-2*r + a**2 + r**2)*(-8*a**3*r*(-2*r + a**2 + r**2)*(2*a**2*r*cos(th)**2 + 2*r**3 - a**4*cos(th)**2 - a**2*r**2*cos(th)**2 - a**2*r**2 - r**4)*sin(th)**3*cos(th)*xph*xth + 4*a*r*(-2*r + a**2 + r**2)*(2*r**2*(a**2 + r**2) - (a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (- r)*(a**2*cos(th)**2 + r**2)**2)*sin(th)**2*xph*xr + 4*a*(r*(2*r**2*(a**2 + r**2) - (a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (- r)*(a**2*cos(th)**2 + r**2)**2) + (a**2*cos(th)**2 - r**2)*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2)))*(-2*r + a**2 + r**2)*sin(th)**2*xph*xr - 1*(a**2*cos(th)**2 - r**2)*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2))*(-2*r + a**2 + r**2)*xt**2 + 2*a**2*(16**2*r**2*(a**2 + r**2)*(-2*r + a**2 + r**2) + (a**2*cos(th)**2 + r**2)**2*(-2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(2*r - 1*a**2 - 1*r**2)))*sin(th)*cos(th)*xr*xth + (2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2))*(-2*r + 1*a**2 + 1*r**2)*(2*r**2*(a**2 + r**2) - (a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (- r)*(a**2*cos(th)**2 + r**2)**2)*sin(th)**2*xph**2))/((a**2*cos(th)**2 + r**2)*(16**2*a**2*r**2*(-2*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2)))*(-2*r + a**2 + r**2))
    #theta'
    f3 = -(-1*a**2*r*(-2*r + a**2 + r**2)*sin(2*th)*xt**2 - 0.5*a**2*(a**2*cos(th)**2 + r**2)**2*(-2*r + a**2 + r**2)*sin(2*th)*xth**2 + 0.5*a**2*(a**2*cos(th)**2 + r**2)**2*sin(2*th)*xr**2 + 2*r*(a**2*cos(th)**2 + r**2)**2*(-2*r + a**2 + r**2)*xr*xth - (2*a**2*r*(a**2 + r**2)*sin(th)**2 + 1*(a**2*cos(th)**2 + r**2)*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2)))*(-2*r + a**2 + r**2)*sin(th)*cos(th)*xph**2)/((a**2*cos(th)**2 + r**2)**3*(-2*r + a**2 + r**2))
    #phi'
    f4 = -(-8*a*r*(a**2*cos(th)**2 + r**2)**2*(-2*r + a**2 + r**2)*(a**2*sin(th)**2 + a**2 + r**2)*xr*xth + 4*a*(a**2*cos(th)**2 + r**2)**2*(r*(r*(-2*r + a**2 + r**2) + (- r)*(a**2*cos(th)**2 + r**2)) + (a**2*cos(th)**2 - r**2)*(2*r - a**2 - r**2))*tan(th)*xr**2 - (a**2*cos(th)**2 + r**2)**2*(4*a*r**2*(-2*r + a**2 + r**2)*xth**2 + (2*r**2*(a**2 + r**2) - (1*a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (- r)*(a**2*cos(th)**2 + r**2)**2)*xph*xr)*(-2*r + a**2 + r**2)*tan(th) + 2*(16**2*a**2*r**2*(a**2 + r**2)*(-2*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(2*a**2*r*(a**2 + r**2)*sin(th)**2 + 1*(a**2*cos(th)**2 + r**2)*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2))))*(-2*r + a**2 + r**2)*xph*xth + (-2*r + a**2 + r**2)*(-4**2*a*r*(a**2*cos(th)**2 - r**2)*(-2*r + a**2 + r**2)*xt**2 + 4*a*r*(-2*r + a**2 + r**2)*(2*r**2*(a**2 + r**2) - (a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (- r)*(a**2*cos(th)**2 + r**2)**2)*sin(th)**2*xph**2 + (16**2*a**2*r*(a**2*cos(th)**2 - r**2)*(-2*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(-2*r**2*(a**2 + r**2) + (a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + r*(a**2*cos(th)**2 + r**2)**2))*xph*xr)*tan(th))/((a**2*cos(th)**2 + r**2)*(16**2*a**2*r**2*(-2*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(2*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*r + a**2 + r**2)))*(-2*r + a**2 + r**2)*tan(th))
    f=[f1,xt,f2,xr,f3,xth,f4,xph]
    return f

# ODE solver parameters
stop = 40
numpoints = 400

# Form time basis
lamb = linspace(0,stop,numpoints)

def event(lamb,X,P):
    return abs(X[0])-abs(1+(1-P[0]**2)**(1/2))
event.terminal=True

sol = solve_ivp(mydiff, [0,stop], varlist, method='LSODA', args=(constlist,), events=[event])

r=sol.y[3]
theta=sol.y[5]
phi=sol.y[7]


fig = plt.pyplot.figure()
ax = plt.pyplot.gca(projection='3d') 
ax.plot(r*cos(phi)*sin(theta),r*cos(theta)*sin(phi),r*cos(theta))
#ax.plot([-6.5],[0],[0])
#ax.plot([0],[-6.5],[0])
#ax.plot([0],[6.5],[0])
#ax.plot([0],[0],[-6.5])
#ax.plot([0],[0],[6.5])

u, v = mgrid[0:2*pi:50j, 0:pi:50j]
x = (1+(1-constlist[0]**2)**(1/2))*sin(v)*cos(u)
y = (1+(1-constlist[0]**2)**(1/2))*sin(v)*sin(u)
z = (1+(1-constlist[0]**2)**(1/2))*cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=1)
ax.azim = 50
ax.elev = 30
plt.pyplot.show()