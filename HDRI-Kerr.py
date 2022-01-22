# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:47:23 2022

@author: jswee
"""

from numpy import pi, sin, cos, sqrt, arctan, arccos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from PIL import Image

img=Image.open("C:/Users/jswee/OneDrive/Desktop/Project/free_star_sky_hdri.jpg")
imageCopy=Image.open("C:/Users/jswee/OneDrive/Desktop/Project/free_star_sky_hdri.jpg")

width=img.size[0]
height=img.size[1]
halfWidth=int(width/2)
halfHeight=int(height/2)
celestialr = 200

r, theta, phi = [100, pi/2, 0]
v = 1.001
a = 0.99
x0 = sqrt(r**2+a**2)*sin(theta)*cos(phi)
y0 = sqrt(r**2+a**2)*sin(theta)*sin(phi)
z0 = r*cos(theta)
counter1=0
counter2=0
for j in range(height):
    #for i in range(-int(((halfHeight)**2-(j-halfHeight)**2)**(1/2)),int(((height/2)**2-(j-halfHeight)**2)**(1/2))):
    for i in range(width):
        counter1+=1
        if counter1%int(height*width/100)==0:
            counter2+=1
            counter1=0
            print(counter2)
        x=(i-halfWidth)*pi/height
        y=(halfHeight-j)*pi/height
        alpha = arccos(cos(y)*cos(x))
        if x>0:
            beta=arctan(y/x)
        elif x<0:
            beta=pi+arctan(y/x)
        if x==0:
            if y>0:
                beta=pi/2
            else:
                beta=3*pi/2
        launch = 3*pi/2 - alpha
        incl = beta - pi/2
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
        varlist = [0, r, theta, phi, pr, ptheta]
        
        
        # ODE solver parameters
        
        def event(tau, X):
            return abs(X[1]) - abs(1 + (1 - (a*cos(X[2]))**2)**(1/2))
        event.terminal=True
        
        def mydiff(tau, X=varlist):
            t, r, theta, phi, pr, ptheta = X
            td = E + (2*r*(r**2 + a**2)*E - 2*a*r*L)/(Sigma(r, theta)*Delta(r))
            rd = pr*Delta(r)/Sigma(r, theta)
            thetad = ptheta/Sigma(r, theta)
            phid = (2*a*r*E + (Sigma(r, theta) - 2*r)*L/(sin(theta)**2))/(Sigma(r, theta)*Delta(r))
            prd = 1/(Sigma(r, theta)*Delta(r))*(-k*(r - 1)+2*r*(r**2 + a**2)*E**2 - 2*a*E*L) - (2*pr**2*(r - 1))/Sigma(r, theta)
            pthetad = (sin(theta)*cos(theta))/(Sigma(r, theta))*(L**2/(sin(theta)**4) - (a*E)**2)
            return [td, rd, thetad, phid, prd, pthetad]
        
        sol = solve_ivp(mydiff, [0,celestialr], varlist, events=[event])

        rlst = sol.y[1]
        thetalst=sol.y[2]
        philst=sol.y[3]
        
        if sol.status == 1:
            img.putpixel((i,j),(0,0,0))
        else:
            x1 = sqrt(rlst[-1]**2+a**2)*sin(thetalst[-1])*cos(philst[-1])
            y1 = sqrt(rlst[-1]**2+a**2)*sin(thetalst[-1])*sin(philst[-1])
            z1 = rlst[-1]*cos(thetalst[-1])
            if y1>y0:
                endtheta = arctan((x1-x0)/(y1-y0))
                endphi = arctan((z1-z0)/(sqrt((x1-x0)**2+(y1-y0)**2)))
            elif y1<y0:
                endtheta = pi + arctan((x1-x0)/(y1-y0))
                endphi = arctan((z1-z0)/(sqrt((x1-x0)**2+(y1-y0)**2)))
            endpixx = halfHeight+int(endtheta*height/pi)
            endpixy = halfHeight-int(endphi*height/pi)
            if endpixx<0:
                endpixx=0
            if endpixx>width-1:
                endpixx=width-1
            if endpixy<0:
                endpixy=0
            if endpixy>height-1:
                endpixy=height-1
            else:
                #print(i+halfWidth,j+halfHeight)
                #print(endpixx,endpixy)
                #print(height,width)
                img.putpixel((i,j),imageCopy.getpixel((endpixx,endpixy)))
            
figure, ax = plt.subplots()
ax.axis("equal")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
img.save("C:/Users/jswee/OneDrive/Desktop/Project/free_star_sky_hdri-kerr-full-new.jpg")        

plt.imshow(img)
plt.show()