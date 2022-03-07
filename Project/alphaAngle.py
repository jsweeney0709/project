# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:42:28 2021

@author: jswee
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects


# State ODE as coupled first order
def mydiff(t, X, P):
    M, L = P
    r, s, phi = X
    f = [s, -L**2*(3*M-r)/(r**4), L/r**2]
    return f

# ODE solver parameters
stoptime = 5000
numpoints = 700


X0 = [100, -1, np.pi]
P = [4, 0]

#Terminate if ray hits the event horizon - for some reason doesn't work for alpha=0
def event(t,X,P):
    return abs(2*P[0])-abs(X[0])
event.terminal=True

# Form time basis
t = np.linspace(0,stoptime,numpoints)


stepsize=50
bigstepsize=150
biggest=[]
newbiggest=[]
for i in range(stepsize):
    # Initial conditions
    alpha=i*(np.pi/(2*stepsize))
    L=(X0[0]*X0[1]*np.cos(X0[2])*np.tan(alpha)-X0[0]*X0[1]*np.sin(X0[2]))/(np.cos(X0[2])+np.sin(X0[2])*np.tan(alpha))
    P[1] = L

    # Solve
    sol = solve_ivp(mydiff, [0,stoptime], X0, args=(P,), method='LSODA', events=[event], t_eval=t)
    
    if sol.t_events[0].size>0:
        biggest=alpha
        biggestI=i
        
    if biggest and sol.t_events[0].size==0:
        break
    
for j in range(bigstepsize):
    # Initial conditions
    alpha=biggest+j*((biggestI+1)*(np.pi/(2*stepsize))-biggest)/(bigstepsize)
    L=(X0[0]*X0[1]*np.cos(X0[2])*np.tan(alpha)-X0[0]*X0[1]*np.sin(X0[2]))/(np.cos(X0[2])+np.sin(X0[2])*np.tan(alpha))
    P[1] = L

    # Solve
    sol = solve_ivp(mydiff, [0,stoptime], X0, args=(P,), method='LSODA', events=[event], t_eval=t)
    if sol.t_events[0].size>0:
        newbiggest=alpha
    if newbiggest and sol.t_events[0].size==0:
        break
        

plotListY=[]
plotListX=[]
lList=np.linspace(newbiggest,1.5,400)
leval=np.linspace(newbiggest,1.5,800)


def event(t,X,P):
    return abs(2*P[0])-abs(X[0])
event.terminal=True

# Solve
for i in lList:
    X0 = [100, -1, np.pi]
    alpha=i
    L=(X0[0]*X0[1]*np.cos(X0[2])*np.tan(alpha)-X0[0]*X0[1]*np.sin(X0[2]))/(np.cos(X0[2])+np.sin(X0[2])*np.tan(alpha))
    P[1]=L
    sol = solve_ivp(mydiff, [0,stoptime], X0, args=(P,), events=[event])

    a = sol.y[0]
    b = sol.y[2]
            
    xlist=np.linspace(a[0]*np.cos(b[0]),a[-1]*np.cos(b[-1]))
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
            else:
                answer = np.pi-tan
        else:
            if q<0:
                answer = -tan
            else:
                answer = tan
        return answer+np.ceil(k/(2*np.pi))*(2*np.pi)
    if (sol.status==0 and i!=0):
        plotListX.append(i)
        plotListY.append(findAngle(a[0]*np.cos(b[0]),a[0]*np.sin(b[0]),a[-1]*np.cos(b[-1]),a[-1]*np.sin(b[-1])))
    else:
        leval=np.delete(leval,[0,1])
        
class CenteredFormatter(mpl.ticker.ScalarFormatter):
    """Acts exactly like the default Scalar Formatter, but yields an empty
    label for ticks at "center"."""
    center = 0
    def __call__(self, value, pos=None):
        if value == self.center:
            return ''
        else:
            return mpl.ticker.ScalarFormatter.__call__(self, value, pos)

figure, ax = plt.subplots()
interpol = np.interp(leval,plotListX,plotListY)
plt.plot(leval,interpol)
plt.plot(plotListX,plotListY)
plt.axis([0, 1.55, -4 , 1.5])
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
formatter = CenteredFormatter()
formatter.center = [0,0]
ax.set_major_formatter(formatter)
plt.xlabel('Alpha')
plt.ylabel('Alpha prime')
plt.show()
