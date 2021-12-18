# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:37:03 2021

@author: jswee
"""
from numpy import zeros, sin, cos, tan, pi, sqrt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


height=360
width=720
window_height=0.00001
window_width = (width/height)*window_height
distance_from_window=0.000014*(-1)
a = 0.95


def Sigma(r,theta):
    return(a*cos(theta))**2+r**2
def drSigma(r):
    return 2*r
def dthetaSigma(r,theta):
    return -2*cos(theta)*sin(theta)*a**2
def Delta(r):
    return a**2+r**2-2*r
def drDelta(r):
    return 2*r-2

#coords(i,j,1) = theta
#coords(i,j,2) = phi
#coords(i,j,3) = 1,0 if stopped at 2,R_celest
coords=zeros(height,width,3)
stepsize=0.1
for j in range(1,width+1):
    for i in range(1,height+1):
        h=window_height/2-(i-1)*window_height/(height-1)
        w=-window_width/2+(j-1)*window_width/(width-1)
        initial = [70,pi/2-pi/46,0]
        r,theta,phi = [initial]
        t_dot=1
        phi_dot=w/(sin(theta)*sqrt(a**2+r**2)*(distance_from_window**2+w**2+h**2))
        p_r=2*Sigma(r,theta)*(h*(a**2+r**2)*cos(theta)+r*sqrt(a**2+r**2)*sin(theta)*distance_from_window)/(sqrt(distance_from_window**2+h**2+w**2)*(a**2+2*r**2+cos(2*theta)*a**2)*Delta(r))
        p_theta=2*Sigma(r,theta)*(-h*r*sin(theta)+sqrt(a**2+r**2)*cos(theta)*distance_from_window)/(sqrt(distance_from_window**2+h**2+w**2)*(a**2+2*r**2+cos(2*theta)*a**2))
        
        E=(1-2/r)*t_dot+(2*a*phi_dot)/r
        L=-2*a*t_dot/r*+(r**2+a**2+(2*a**2)/r)*phi_dot
        
        def f(lambd,x):
            #x=[r,theta,phi,pr,ptheta]
            f1=(x[3]*Delta(x[0]))/Sigma(x[0],x[1])
            f2=x[4]/Sigma(x[0],x[1])
            f3=(a*(-a*L+x[0]*2*E)+L*Delta(x[0]/sin(x[1])**2))/(Delta(x[0])*Sigma(x[0],x[1]))
            f4=-1/(2*Sigma(x[0],x[1])*((-E*Delta(x[0]))*(a*2*(-2*L+a*E*sin(x[1])**2)+2*x[0]*E*Sigma(x[0],x[1]))*(Sigma(x[0],x[1])*Delta(x[0]))**2)+(a*(a*L**2-2*L*x[0]*2*E+a*x[0]*2*(sin(x[1])*E)**2)+(Delta(x[0])*x[3])**2+(a**2+x[0]**2)*Sigma(x[0],x[1])*E**2)*drDelta(x[0]))
            f5=-1/(2*Delta(x[0])*Sigma(x[0],x[1])**2)*(-2*sin(x[1])*(a**2*x[0]*2*cos(x[1])*E**2+(Delta(x[0])*L**2)/(tan(x[1])*sin(x[1])**3)*Sigma(x[0],x[1])+(a*(L*(a*L-2*x[0]*2*E)+a*x[0]*2*(sin(x[1])*E)**2)-Delta(x[0])*(x[4]**2+(L**2/sin(x[1])**2)+Delta(x[0])*x[3]**2)))*dthetaSigma(x[1]))
            return [f1,f2,f3,f4,f5]



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
x = Iscop*np.cos(u)*np.sin(v)
y = Iscop*np.sin(u)*np.sin(v)
z = Iscop*np.cos(v)
# alpha controls opacity
ax.plot_surface(x, y, z, color="black", alpha=0.8)
ax.azim = 90
ax.elev = 10
plt.show()

