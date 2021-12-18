import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from random import randint
import tarfile
from numpy import loadtxt, linspace, outer, ones, size
from PIL import Image, ImageChops, ImageOps
from papirus import PapirusImage
from scipy.integrate import odeint
from numpy import sqrt, zeros, pi, linspace, append, loadtxt, pi, abs, sin, cos, tan, sqrt
import os
import time
from random import uniform


def RHS(w, s, p):
    """
    Defines the differential equations for the system 
    found using SymPy and GraviPy.
    Arguments:
        w :  vector of the state variables:
                  x = [t, xr=r', r, xth=th', th, xph=ph', ph]
        s :  affine parameter
        p :  vector of the parameters:
                  p = [rs,M,a]
    """
    xt, t, xr, r, xth, th, xph, ph = w
    rs, M, a = p
    rho = r**2 + a**2 * cos(th)**2
    Delta = r**2 - 2*M*r + a**2
    xt = sqrt(-(Delta*rho*xph**2*sin(th)**2 + 2*M*a**2*r*xph**2*sin(th)**2 + 2*M*r**3*xph**2*sin(th)**2 + rho**2*xth**2 + rho**2*xr**2/Delta)/(2*M*r - rho))
    
    geodesic = [0,0,0,0]
    geodesic[0] = 2.0*M*(a**2*r*sin(2*th)*xth + (a**2*cos(th)**2 - r**2)*xr)*xt/((a**2*cos(th)**2 + r**2)*(-2*M*r + a**2*cos(th)**2 + r**2))

    geodesic[1] = -(-1.0*r*(a**2*cos(th)**2 + r**2)**2*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2))*(-2*M*r + a**2 + r**2)**2*xth**2 + (16.0*M**2*a**2*r*(a**2*cos(th)**2 - r**2)*(-2*M*r + a**2 + r**2)**2*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(r*(-2*M*r + a**2 + r**2) + (M - r)*(a**2*cos(th)**2 + r**2))*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2)))*xr**2 + (-2*M*r + a**2 + r**2)*(-8.0*M*a**3*r*(-2*M*r + a**2 + r**2)*(2*M*a**2*r*cos(th)**2 + 2*M*r**3 - a**4*cos(th)**2 - a**2*r**2*cos(th)**2 - a**2*r**2 - r**4)*sin(th)**3*cos(th)*xph*xth + 4.0*M*a*r*(-2*M*r + a**2 + r**2)*(2*M*r**2*(a**2 + r**2) - M*(a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (M - r)*(a**2*cos(th)**2 + r**2)**2)*sin(th)**2*xph*xr + 4.0*M*a*(r*(2*M*r**2*(a**2 + r**2) - M*(a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (M - r)*(a**2*cos(th)**2 + r**2)**2) + (a**2*cos(th)**2 - r**2)*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2)))*(-2*M*r + a**2 + r**2)*sin(th)**2*xph*xr - 1.0*M*(a**2*cos(th)**2 - r**2)*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2))*(-2*M*r + a**2 + r**2)*xt**2 + 2*a**2*(16.0*M**2*r**2*(a**2 + r**2)*(-2*M*r + a**2 + r**2) + (a**2*cos(th)**2 + r**2)**2*(-2.0*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(2.0*M*r - 1.0*a**2 - 1.0*r**2)))*sin(th)*cos(th)*xr*xth + (2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2))*(-2.0*M*r + 1.0*a**2 + 1.0*r**2)*(2*M*r**2*(a**2 + r**2) - M*(a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (M - r)*(a**2*cos(th)**2 + r**2)**2)*sin(th)**2*xph**2))/((a**2*cos(th)**2 + r**2)*(16*M**2*a**2*r**2*(-2*M*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2)))*(-2*M*r + a**2 + r**2))

    geodesic[2] = -(-1.0*M*a**2*r*(-2*M*r + a**2 + r**2)*sin(2*th)*xt**2 - 0.5*a**2*(a**2*cos(th)**2 + r**2)**2*(-2*M*r + a**2 + r**2)*sin(2*th)*xth**2 + 0.5*a**2*(a**2*cos(th)**2 + r**2)**2*sin(2*th)*xr**2 + 2.0*r*(a**2*cos(th)**2 + r**2)**2*(-2*M*r + a**2 + r**2)*xr*xth - (2.0*M*a**2*r*(a**2 + r**2)*sin(th)**2 + 1.0*(a**2*cos(th)**2 + r**2)*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2)))*(-2*M*r + a**2 + r**2)*sin(th)*cos(th)*xph**2)/((a**2*cos(th)**2 + r**2)**3*(-2*M*r + a**2 + r**2))

    geodesic[3] = -(-8.0*M*a*r*(a**2*cos(th)**2 + r**2)**2*(-2*M*r + a**2 + r**2)*(a**2*sin(th)**2 + a**2 + r**2)*xr*xth + 4.0*M*a*(a**2*cos(th)**2 + r**2)**2*(r*(r*(-2*M*r + a**2 + r**2) + (M - r)*(a**2*cos(th)**2 + r**2)) + (a**2*cos(th)**2 - r**2)*(2*M*r - a**2 - r**2))*tan(th)*xr**2 - (a**2*cos(th)**2 + r**2)**2*(4.0*M*a*r**2*(-2*M*r + a**2 + r**2)*xth**2 + (2.0*M*r**2*(a**2 + r**2) - M*(1.0*a**2 + 3.0*r**2)*(a**2*cos(th)**2 + r**2) + 1.0*(M - r)*(a**2*cos(th)**2 + r**2)**2)*xph*xr)*(-2*M*r + a**2 + r**2)*tan(th) + 2*(16.0*M**2*a**2*r**2*(a**2 + r**2)*(-2*M*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(2.0*M*a**2*r*(a**2 + r**2)*sin(th)**2 + 1.0*(a**2*cos(th)**2 + r**2)*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2))))*(-2*M*r + a**2 + r**2)*xph*xth + (-2*M*r + a**2 + r**2)*(-4.0*M**2*a*r*(a**2*cos(th)**2 - r**2)*(-2*M*r + a**2 + r**2)*xt**2 + 4.0*M*a*r*(-2*M*r + a**2 + r**2)*(2*M*r**2*(a**2 + r**2) - M*(a**2 + 3*r**2)*(a**2*cos(th)**2 + r**2) + (M - r)*(a**2*cos(th)**2 + r**2)**2)*sin(th)**2*xph**2 + (16.0*M**2*a**2*r*(a**2*cos(th)**2 - r**2)*(-2*M*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(-2.0*M*r**2*(a**2 + r**2) + M*(1.0*a**2 + 3.0*r**2)*(a**2*cos(th)**2 + r**2) + 1.0*(-M + r)*(a**2*cos(th)**2 + r**2)**2))*xph*xr)*tan(th))/((a**2*cos(th)**2 + r**2)*(16*M**2*a**2*r**2*(-2*M*r + a**2 + r**2)*sin(th)**2 + (a**2*cos(th)**2 + r**2)**2*(2*M*r*(a**2 + r**2) + (a**2*cos(th)**2 + r**2)*(-2*M*r + a**2 + r**2)))*(-2*M*r + a**2 + r**2)*tan(th))

    # Create f = w' = (xt',t',xr',r',xth',th',xph',ph'):
    f = [geodesic[0],
         xt,
         geodesic[1],
 	 xr,
	 geodesic[2],
	 xth,
	 geodesic[3],
	 xph]
    return f

def writeToFile(proper_time,solutions,p,endPoint):
    """
    Write the solutions to a file
    """
    with open('output.dat', 'w') as f:
        r1 = solutions[0][3]
        th1 = solutions[0][5]
        ph1 = solutions[0][7]
        xth1 = solutions[0][4]
        xph1 = solutions[0][6]
        f.write('# '+' '.join([str(i) for i in p])+' '+str(r1)+' '+str(endPoint)+'\n')
        # Print & save the solution.
        for time, solution in zip(proper_time, solutions):
            f.write(str(time)+' '.join([" %s" % i for i in solution])+'\n')    
    # tarName = 'data/r1='+'%4.2f'%r1 + '-th1='+'%4.3f'%th1 + '-ph1='+'%4.3f'%ph1 + '-xth1='+'%4.3f'%xth1 + '-xph1='+'%4.3f'%xph1+'.tar.gz'
    # make_tarfile(tarName, 'output.dat')

def solveGeodesic(RightHandSide,s,p,x0):
    """
    Solve the equations. We need to check at each time step whether the particle is too close to the black hole or has escaped.
    Define escaped as 10% further away from the BH than it started. This makes no sense if we start close to the BH.
    """
    solutions = [[0,0,0,0] for i in range(numpoints)]
    solutions[0] = x0
    rs,M,a = p

    # ODE solver parameters
    abserr = 1.0e-6
    relerr = 1.0e-6

    # endPoint is 0 if the ray escapes and 1 if it falls in. keep this in case I ever want to draw the shadow
    endPoint = 0
    # loop each time step, kill loop if r is too close to rs
    for idx in range(1,int(numpoints)):
        if x0[3]>1.01*rs and x0[3]<1.1*r1:
            solutions[idx] = odeint(RightHandSide, x0, [s[idx-1],s[idx]], args=(p,), atol=abserr, rtol=relerr)[1]
            x0 = solutions[idx]
        else:
            if x0[3]<=1.05*rs:
                endPoint = 1
            if x0[3]>=5*r1:
                endPoint = 0
            writeToFile(s[:idx],solutions[:idx],p,endPoint)
            break
        #time.sleep(0.1)
    return s[:idx], solutions[:idx]

def run(RHS, s, p, x0):
    s, xsol=solveGeodesic(RHS,s,p,x0)
    return s, xsol

def make_tarfile(output_filename, source_dir):
    """
    tar the solution and save it
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

# Use ODEINT to solve the differential equations defined by the vector field

tstart = time.time()

# Parameter values. these provide the initial velocities via line 11
M = 1
a = 0.5
rs = M+sqrt(M**2-a**2)
rs = 2*M

# Initial conditions
# randomize the initial position
s1 = 0
r1 = uniform(15,25) # in multiples of rs
# th1 = pi/2.
# ph1 = pi/2. # this is where "around" the BH the ray starts, 0 for east, pi/2 for north, etc
th1 = uniform(0,pi)
ph1 = uniform(0,2*pi)

xr1 = -0.1 # this is initial r velocity if you'd like
# here below are tests for the angles
# xth1 = 0.001 # up above the BH and then shoots below items
# xth1 = -0.001 # down below the BH and then shoots up above it
# xph1 = -0.001 # a bit up
# xph1 = 0 # straight in
# xph1 = 0.001 # a bit down

# randomize the initial angles but always shoot towards the black hole (ish)
xth1 = randint(-1,1)*uniform(0.0005,0.001)
xph1 = randint(-1,1)*uniform(0.0005,0.001)


# affine parameter values
sf = 10**4
numpoints = sf*20+1
s = linspace(s1,sf,numpoints)

# Pack up the parameters and initial conditions:
p = [rs,M,a]
x0 = [xt1,s1,xr1,r1,xth1,th1,xph1,ph1]

# loop over b's
#for i in linspace(-3,1,31):
#    run(RHS, s, [rs, a, i], x0)
run(RHS, s, p, x0)
tend = time.time()
print(tend-tstart)

def draw3D(M,a,r1):
    s, xt, t, xr, r, xth, th, xph, ph = loadtxt('output.dat', unpack=True)
    fline=open('output.dat').readline().rstrip().split()
    rs = float(fline[1])
    M = float(fline[2])
    a = float(fline[3])
    r1 = float(fline[4])
    th1 = float(th[1])
    ph1 = float(ph[1])
    rmax = max(r)

    fig = plt.figure(figsize=(12.8*0.5, 8*0.5))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    ax.axis('off')
    ax.autoscale(enable=False,axis='both')  #you will need this line to change the Z-axis
    ax.set_xbound(-0.3*r1*1.6, 0.3*r1*1.6)
    ax.set_ybound(-0.3*r1*1.6, 0.3*r1*1.6)
    ax.set_zbound(-0.3*r1, 0.3*r1)
    x = r * cos(th) * sin(ph)
    y = r * sin(th) * sin(ph)
    z = r * cos(ph)
    
    ax.plot(x, y, z, color='black', linewidth=1.5)
    
    X,Y,Z = generateSphere(rs,[0,0,0])
    ax.plot_surface(X, Y, Z,  color='black', linewidth=0, alpha=1)
    ax.view_init(elev=randint(-90,90), azim=randint(0,360))
    ax.view_init(elev=0, azim=0)
    plt.savefig('output.png',dpi=800)

def invertImage():
    image = Image.open('output-trimmed.png')
    image = image.convert('L')
    inverted_image = ImageOps.invert(image)
    inverted_image = inverted_image.convert('1')
    inverted_image.save('output-inverted.png')


def trim(file):
    img = Image.open(file+'.png')
    w, h = img.size
    r = 1280./800
    x = int((h-w/r)/2)
    final_img = img.crop((0, x, w, h-x))
    final_img.save(file+'-trimmed.png')

def drawOnScreen():
    image = PapirusImage()
    image.write('output-inverted.png')

def generateSphere(r, c):
    u = linspace(0, 2 * pi, 200)
    v = linspace(0, pi, 200)
    x_c, y_c, z_c = c
    X = r * outer(cos(u), sin(v)) + x_c
    Y = r * outer(sin(u), sin(v)) + y_c 
    Z = r * outer(ones(size(u)), cos(v)) + z_c
    return X,Y,Z

M=1
a=0.5
r1=10
# drawBitmap(M,a,r1)
draw3D(M,a,r1)
# drawOnScreen()
