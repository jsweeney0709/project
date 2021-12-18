# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:24:45 2021

@author: jswee
"""
from numpy import sin, cos, tan, pi, sqrt, mgrid
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


height=3600
width=7200
window_height=0.00001
window_width = (width/height)*window_height
distance_from_window=0.000014*(-1)
a = 0.95


def Sigma(r,theta):
    return(a*cos(theta))**2+r**2
def drSigma(r):
    return 2*r
def dthetaSigma(theta):
    return -2*cos(theta)*sin(theta)*a**2
def Delta(r):
    return a**2+r**2-2*r
def drDelta(r):
    return 2*r-2

#coords(i,j,1) = theta
#coords(i,j,2) = phi
#coords(i,j,3) = 1,0 if stopped at 2,R_celest
#coords=zeros(height,width,3)
stepsize=0.1

j=height
i=height

h=window_height/2-(i-1)*window_height/(height-1)
w=-window_width/2+(j-1)*window_width/(width-1)
initial = [20,pi/2,0]
r,theta,phi = initial
t_dot=1
phi_dot=w/(sin(theta)*sqrt(a**2+r**2)*(distance_from_window**2+w**2+h**2))
p_r=2*Sigma(r,theta)*(h*(a**2+r**2)*cos(theta)+r*sqrt(a**2+r**2)*sin(theta)*distance_from_window)/(sqrt(distance_from_window**2+h**2+w**2)*(a**2+2*r**2+cos(2*theta)*a**2)*Delta(r))
p_theta=2*Sigma(r,theta)*(-h*r*sin(theta)+sqrt(a**2+r**2)*cos(theta)*distance_from_window)/(sqrt(distance_from_window**2+h**2+w**2)*(a**2+2*r**2+cos(2*theta)*a**2))

varlist=[r,theta,phi,p_r,p_theta]


E=(1-2/r)*t_dot+(2*a*phi_dot)/r
L=-2*a*t_dot/r*+(r**2+a**2+(2*a**2)/r)*phi_dot

lst=[]

def mydiff(lambd,x):
    #x=[r,theta,phi,pr,ptheta]
    f1=(x[3]*Delta(x[0]))/Sigma(x[0],x[1])
    f2=x[4]/Sigma(x[0],x[1])
    lst.append(f2)
    f3=(a*(-a*L+x[0]*2*E)+L*Delta(x[0]/sin(x[1])**2))/(Delta(x[0])*Sigma(x[0],x[1]))
    f4=-1/(2*Sigma(x[0],x[1])*((-E*Delta(x[0]))*(a*2*(-2*L+a*E*sin(x[1])**2)+2*x[0]*E*Sigma(x[0],x[1]))*(Sigma(x[0],x[1])*Delta(x[0]))**2)+(a*(a*L**2-2*L*x[0]*2*E+a*x[0]*2*(sin(x[1])*E)**2)+(Delta(x[0])*x[3])**2+(a**2+x[0]**2)*Sigma(x[0],x[1])*E**2)*drDelta(x[0]))
    f5=-1/(2*Delta(x[0])*Sigma(x[0],x[1])**2)*(-2*sin(x[1])*(x[0]*2*cos(x[1])*(E*a)**2+(Delta(x[0])*L**2)/(tan(x[1])*sin(x[1])**3)*Sigma(x[0],x[1])+(a*(L*(a*L-2*x[0]*2*E)+a*x[0]*2*(sin(x[1])*E)**2)-Delta(x[0])*(x[4]**2+(L**2/sin(x[1])**2)+Delta(x[0])*x[3]**2)))*dthetaSigma(x[1]))
    return [f1,f2,f3,f4,f5]

stop = 30

#Define Isco
Z1 = 1+((1-a**2)**(1/3))*((1+a)**(1/3)+(1-a)**(1/3))
Z2 = (3*a**2+Z1**2)**(1/2)
Iscop = 3+Z2+((3-Z1)*(3+Z1+2*Z2))**(1/2)
Iscom = 3+Z2-((3-Z1)*(3+Z1+2*Z2))**(1/2)

def event(t,X):
    return abs(X[0])-abs(1+(1-a**2)**(1/2))
event.terminal=True

sol = solve_ivp(mydiff, [0,stop], varlist, events=[event])

r=sol.y[0]
theta=sol.y[1]
phi=sol.y[2]

fig = plt.figure()
ax = plt.gca(projection='3d') 
ax.plot(r*cos(phi)*sin(theta),r*cos(theta)*sin(phi),r*cos(theta))
ax.plot([-10],[0],[0])
ax.plot([10],[0],[0])
ax.plot([0],[-10],[0])
ax.plot([0],[10],[0])
ax.plot([0],[0],[-10])
ax.plot([0],[0],[10])

u, v = mgrid[0:2*pi:50j, 0:pi:50j]
x = (1+(1-a**2)**(1/2))*cos(u)*sin(v)
y = (1+(1-a**2)**(1/2))*sin(u)*sin(v)
z = (1+(1-a**2)**(1/2))*cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=1)
ax.azim = 90
ax.elev = 00
plt.show()
