# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:31:02 2021

@author: jswee
"""

from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
    
img=Image.open("C:/Users/jswee/OneDrive/Desktop/Project/star-field.jpg")# e.g. C:/Users..../stars.jpg
imageCopy=Image.open("C:/Users/jswee/OneDrive/Desktop/Project/star-field.jpg")


# State ODE as coupled first order
def mydiff(t, X, P):
    M, L = P
    r, s, phi = X
    f = [s, -L**2*(3*M-r)/(r**4), L/r**2]
    return f

# ODE solver parameters
stoptime = 1000
numpoints = 2000


X0 = [100, -1, np.pi]
P = [4, 0]

#Terminate if ray hits the event horizon - for some reason doesn't work for alpha=0
def event(t,X,P):
    return abs(2*P[0])-abs(X[0])
event.terminal=True

# Form time basis
t = np.linspace(0,stoptime,numpoints)

#find critical angle
stepsize=50
bigstepsize=200
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

#angle's we're looking at
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
    
    #check if light orbits
    k=0
    for j in range(1,len(b)):
        k+=b[j]-b[j-1]
    
    #find angle at the end of path to camera
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
    
    #if the light didn't cross the event horizon add angle sent and final angle to lists
    #this gives us a direct relationship between any tangent angle to where it ends
    if (sol.status==0 and i!=0):
        plotListX.append(i)
        plotListY.append(findAngle(a[0]*np.cos(b[0]),a[0]*np.sin(b[0]),a[-1]*np.cos(b[-1]),a[-1]*np.sin(b[-1])))
    else:
        leval=np.delete(leval,[0,1])

#Can make higher quality image slower if you precompute plotListX,plotListY and use np.interp
#for each individual pixel
interpol = np.interp(leval,plotListX,plotListY)

#function to find the index of the closest element in a list to a value
def closest_value(input_list, input_value):
  arr = np.asarray(input_list)
  i = (np.abs(arr - input_value)).argmin()
  return i

width=img.size[0]
height=img.size[1]
halfWidth=int(width/2)
halfHeight=int(height/2)

#for each pixel in the image look at a ray being sent to that source with a black hole at the origin
for i in range(width):
    for j in range(height):
        x=i-halfWidth
        y=j-halfHeight
        if x!=0:
            p=(x/abs(x))*(1.5/((halfWidth**2+halfHeight**2)**(1/2)))*(x**2+y**2)**(1/2)
        if x==0:
            p=(1.5/((halfWidth**2+halfHeight**2)**(1/2)))*(x**2+y**2)**(1/2)
        #use above relation to see where the ray ends up (aprime)
        if p<-newbiggest:
            closeVal=closest_value(leval, -p)
            aprime=-interpol[closeVal]
            
        if p>newbiggest:
            closeVal=closest_value(leval, p)
            aprime=interpol[closeVal]
        #update new image if it ends inside the bounds
        if p<-newbiggest:
            iprime=int((aprime*(x/p))+halfWidth)
            jprime=int((aprime*(y/p))+halfHeight)
            if iprime<width and jprime<height and iprime>-1 and jprime>-1:
                img.putpixel((i,j),imageCopy.getpixel((iprime,jprime)))
            else:
                img.putpixel((i,j),(0,0,0))
        elif p>newbiggest:
            iprime=int((aprime*(x/p))+halfWidth)
            jprime=int((aprime*(y/p))+halfHeight)
            if iprime<width and jprime<height and iprime>-1 and jprime>-1:
                img.putpixel((i,j),imageCopy.getpixel((iprime,jprime)))
            else:
                img.putpixel((i,j),(0,0,0))
        else:
            img.putpixel((i,j),(0,0,0))
            
figure, ax = plt.subplots()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

#save image in higher quality if you want
#img.save("...")        

plt.imshow(img)
plt.show()
