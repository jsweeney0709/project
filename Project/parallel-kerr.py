# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:47:23 2022

@author: jswee
"""

from numpy import pi, sin, cos, sqrt, arctan, arccos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed

img=Image.open("C:/Users/jswee/OneDrive/Desktop/Project/milky-way-hdri-copy.png")
imageCopy=Image.open("C:/Users/jswee/OneDrive/Desktop/Project/milky-way-hdri-copy.png")

width=img.size[0]
height=img.size[1]
halfWidth=int(width/2)
halfHeight=int(height/2)
celestialr = 100

r, theta, phi = [7.5, pi/2, 0]
v = 1
a = 1
x0 = sqrt(r**2+a**2)*sin(theta)*cos(phi)
y0 = sqrt(r**2+a**2)*sin(theta)*sin(phi)
z0 = r*cos(theta)

def solvefor(variable):
    i=variable%width
    j=int((variable-i)/width)
    x=(i-halfWidth)*pi/height
    y=(halfHeight-j)*pi/height
    
    alpha = 3*pi/2+arccos(cos(y)*cos(x))
    
    if x==0:
        if y>0:
            beta=pi
        else:
            beta=pi+pi
    elif y==pi/2 or y==-pi/2:
        alpha=y
        beta=0
    #elif y==0:
        #   alpha=x
        #  beta=0
    elif x==pi or x==-pi:
        if y>0:
            alpha = pi-y
            beta=pi/2
        else:
            alpha = pi+y
            beta=pi/2
    else:
        if x>0:
            beta = pi/2+arctan(sin(y)/(sin(x)*cos(y)))
        else:
            beta = pi/2+pi+arctan(sin(y)/(sin(x)*cos(y)))
        
    launch = alpha
    incl = beta
    vr = v*sin(launch)
    vphi = v*cos(launch)*sin(incl)
    vtheta = v*cos(launch)*cos(incl)
    def Sigma(r, theta):
        return (a*cos(theta))**2 + r**2
    def Delta(r):
        return a**2 + r**2 - 2*r
    def Chi(r, theta):
        return (r**2 + a**2)**2 - Delta(r)*(a*sin(theta))**2
    L = vphi*sin(theta)*sqrt(Chi(r, theta)/Sigma(r, theta))
    omega = 2*r*a/Chi(r, theta)
    E = sqrt(Delta(r)*Sigma(r, theta)/Chi(r, theta)) + L*omega
    ptheta = vtheta*sqrt(Sigma(r, theta))
    pr = vr*sqrt((Sigma(r, theta)/Delta(r)))
    Q = ptheta**2 + ((L/sin(theta))**2 - (a*E)**2)*(cos(theta)**2)
    k = Q + L**2 + a**2*E**2
    
    varlist = [r, theta, phi, pr, ptheta]
    
    
    # ODE solver parameters
    eh = 1 + (1 - a**2)**(1/2)
    def event(tau, X):
        return abs(X[1]) - eh
    event.terminal=True
    
    def mydiff(tau, X=varlist):
        r, theta, phi, pr, ptheta = X
        rd = pr*Delta(r)/Sigma(r, theta)
        thetad = ptheta/Sigma(r, theta)
        phid = (2*a*r*E + (Sigma(r, theta) - 2*r)*L/(sin(theta)**2))/(Sigma(r, theta)*Delta(r))
        prd = 1/(Sigma(r, theta)*Delta(r))*(-k*(r - 1)+2*r*(r**2 + a**2)*E**2 - 2*a*E*L) - (2*pr**2*(r - 1))/Sigma(r, theta)
        pthetad = (sin(theta)*cos(theta))/(Sigma(r, theta))*(L**2/(sin(theta)**4) - (a*E)**2)
        return [rd, thetad, phid, prd, pthetad]
    
    sol = solve_ivp(mydiff, [-celestialr,0], varlist, events=[event])
    
    return(sol.status, sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], i, j)
    
answerlst = Parallel(n_jobs=-1, verbose=0, backend="loky")(
             map(delayed(solvefor), range(width*height)))

for item in answerlst:
    status = item[0]
    r=item[1]
    theta=item[2]
    phi=item[3]
    i=item[4]
    j=item[5]
    if status == 1 or abs(r)<2:
        img.putpixel((i,j),(0,0,0))
    else:
        x1 = sqrt(r**2+a**2)*sin(theta)*cos(phi)
        y1 = sqrt(r**2+a**2)*sin(theta)*sin(phi)
        z1 = r*cos(theta)
        if y1>y0:
            endtheta = arctan((x1-x0)/(y1-y0))
            endphi = -arctan((z1-z0)/(sqrt((x1-x0)**2+(y1-y0)**2)))
        elif y1<y0:
            endtheta = pi + arctan((x1-x0)/(y1-y0))
            endphi = -arctan((z1-z0)/(sqrt((x1-x0)**2+(y1-y0)**2)))
        endpixx = int(height+halfHeight+int(endtheta*height/pi))%width
        endpixy = (halfHeight+int(endphi*height/pi))%height
        img.putpixel((i,j),imageCopy.getpixel((endpixx,endpixy)))
            

figure, ax = plt.subplots()
ax.axis("equal")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
img.save("C:/Users/jswee/OneDrive/Desktop/Project/milky-way-hdri-copy-kerr-full-new(1).png")        

plt.imshow(img)
plt.show()