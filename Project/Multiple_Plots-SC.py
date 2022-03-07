# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 12:07:35 2021

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
stoptime = 10000
numpoints = 10000
P = [1, 0]
# Form time basis
t = np.linspace(0,stoptime,numpoints)

# Set up plot
figure, ax = plt.subplots()
ax.axis("equal")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
def plotThese(lst):
    #lst=[[r1,rdot1,phi1,alpha1]]
    
    # Initial conditions
    for i in lst:
        X0 = [i[0], i[1], i[2]]
        alpha=i[3]
        L=(X0[0]*X0[1]*np.cos(X0[2])*np.tan(alpha)-X0[0]*X0[1]*np.sin(X0[2]))/(np.cos(X0[2])+np.sin(X0[2])*np.tan(alpha))
        P[1] = L

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
        
        # Function to help clean code up
        def findx(i):
            return a[i]*np.cos(b[i])
        def findy(i):
            return a[i]*np.sin(b[i])

        # Plot
        plt.plot(a*np.cos(b),a*np.sin(b))
        plt.plot(findx(0),findy(0), 'p')
        

plotThese([[10,-1,np.pi,(k-20)/20] for k in range(41)])        

disk1=plt.Circle((0, 0), 2*P[0], color="k", fill=True)
ax.add_artist(disk1)

# Show
plt.show()