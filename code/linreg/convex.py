import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image as PIL_Image


def diamond(lmbda=1, n=100):
    "get points along diamond at distance lmbda from origin"
    points = []
    x = np.linspace(0, lmbda, num=n // 4)
    points.extend(list(zip(x, -x + lmbda)))

    x = np.linspace(0, lmbda, num=n // 4)
    points.extend(list(zip(x,  x - lmbda)))

    x = np.linspace(-lmbda, 0, num=n // 4)
    points.extend(list(zip(x, -x - lmbda)))

    x = np.linspace(-lmbda, 0, num=n // 4)
    points.extend(list(zip(x,  x + lmbda)))

    return np.array(points)


#print(diamond(lmbda=1, n=40))

w,h = 5,5
beta0 = np.linspace(-w, w, 100)
beta1 = np.linspace(-h, h, 100)
B0, B1 = np.meshgrid(beta0, beta1)

def loss(b0, b1,
         a = 1,
         b = 1,
         c = 0, # axis stretch
         x = -10, # shift x location
         y = 5): # shift y
    return a*(b0-x)**2 + b*(b1-y)**2 + c*b0*b1

def plot_loss(boundary = diamond(lmbda=1, n=100)):
    a = np.random.random()*3
    b = np.random.random()*3
    c = 0
    x = 0
    while np.abs(x)<1:
        x = np.random.random()*2*w - w
    y = 0
    while np.abs(y)<1:
        y = np.random.random()*2*h - h

    fig,ax = plt.subplots(1,1,figsize=(6,6))

    Z = loss(B0,B1,a=a,b=b,c=c,x=x,y=y)
    ax.contour(B0, B1, Z, levels=20, linewidths=.5)

    lmbda = 1
    ax.plot([-w,+w],[0,0], '-', c='k')
    ax.plot([0, 0],[-h,h], '-', c='k')
    ax.plot([0,lmbda,0,-lmbda,0], [lmbda,0,-lmbda,0,lmbda], '-', lw=1.5, c='#A22396')
    ax.scatter([x],[y],s=80, c='k')

    losses = [loss(*edgeloc,a=a,b=b,c=c,x=x,y=y) for edgeloc in boundary]
    # print(losses)
    minloss_idx = np.argmin(losses)
    coeff = boundary[minloss_idx]
    # print(coeff)

    ax.scatter([coeff[0]], [coeff[1]], s=80, c='#D73028')


def animate(ntrials=20, dpi=100, duration=600):
    plt.close()
    for f in glob.glob(f'/tmp/L1-frame-*.png'):
        os.remove(f)

    for i in range(ntrials):
        plot_loss()
        print(f"/tmp/frame-{i}.png")
        plt.savefig(f"/tmp/frame-{i}.png", bbox_inches=0, pad_inches=0, dpi=dpi)
        plt.close()

    images = [PIL_Image.open(image) for image in sorted(glob.glob(f'/tmp/frame-*.png'))]
    images += reversed(images)
    images[0].save(f'/tmp/L1-animation.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=duration,
                   loop=0)
    print("Saved /tmp/L1-animation.gif")

animate()