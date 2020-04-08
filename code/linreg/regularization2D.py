import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image as PIL_Image
from reg_support import *

lmbda = 2
w,h = 10,10
beta0 = np.linspace(-w, w, 100)
beta1 = np.linspace(-h, h, 100)
B0, B1 = np.meshgrid(beta0, beta1)

def select_parameters(lmbda, reg):
    while True:
        a = np.random.random() * 10
        b = np.random.random() * 10
        c = np.random.random() * 4 - 1.5

        # get x,y outside of circle radius lmbda
        x, y = 0, 0
        if reg=='l1':
            while np.abs(x) + np.abs(y) <= lmbda:
                x = np.random.random() * 2 * w - w
                y = np.random.random() * 2 * h - h
        else:
            while np.sqrt(x**2 + y**2) <= lmbda:
                x = np.random.random() * 2 * w - w
                y = np.random.random() * 2 * h - h


        Z = loss(B0, B1, a=a, b=b, c=c, cx=x, cy=y)
        loss_at_min = loss(x, y, a=a, b=b, c=c, cx=x, cy=y)
        if (Z >= loss_at_min).all(): # hooray! we didn't make a saddle point
            break # fake repeat-until in python
        # print("loss not min", loss_at_min)

    return Z, a, b, c, x, y


def plot_loss(boundary, reg, show_contours=True, contour_levels=50, show_loss_eqn=False):
    Z, a, b, c, x, y = select_parameters(lmbda, reg)
    eqn = f"{a:.2f}(b0 - {x:.2f})^2 + {b:.2f}(b1 - {y:.2f})^2 + {c:.2f} b0 b1"

    fig,ax = plt.subplots(1,1,figsize=(6,6))
    if show_loss_eqn:
        ax.set_title(eqn, fontsize=10)
    ax.set_xticks([-10,-5,0,5,10])
    ax.set_yticks([-10,-5,0,5,10])

    if show_contours:
        ax.contour(B0, B1, Z, levels=contour_levels, linewidths=.5, cmap='coolwarm')
    else:
        ax.contourf(B0, B1, Z, levels=contour_levels, cmap='coolwarm')

    # Draw axes
    ax.plot([-w,+w],[0,0], '-', c='k')
    ax.plot([0, 0],[-h,h], '-', c='k')
    ax.plot(boundary[:,0], boundary[:,1], '-', lw=1.5, c='#A22396')

    # Draw center of loss func
    ax.scatter([x],[y],s=80, c='k')

    # Draw point on boundary
    losses = [loss(*edgeloc, a=a, b=b, c=c, cx=x, cy=y) for edgeloc in boundary]
    minloss_idx = np.argmin(losses)
    coeff = boundary[minloss_idx]
    ax.scatter([coeff[0]], [coeff[1]], s=80, c='#D73028')
    # plt.show()


def animate(ntrials=20, reg='l1', dpi=200, duration=600, show_contours=True, contour_levels=50):
    plt.close()
    for f in glob.glob(f'/tmp/{reg}-frame-*.png'):
        os.remove(f)

    if reg=='l1':
        boundary = diamond(lmbda=lmbda, n=100)
    else:
        boundary = circle(lmbda=lmbda, n=100)
    for i in range(ntrials):
        plot_loss(boundary=boundary, reg=reg, show_contours=show_contours, contour_levels=contour_levels)
        print(f"/tmp/{reg}-frame-{i}.png")
        plt.savefig(f"/tmp/{reg}-frame-{i}.png", bbox_inches=0, pad_inches=0, dpi=dpi)
        plt.close()

    images = [PIL_Image.open(image) for image in sorted(glob.glob(f'/tmp/{reg}-frame-*.png'))]
    images += reversed(images)
    images[0].save(f'/tmp/{reg}-animation.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=duration,
                   optimize=False,
                   loop=0)
    print(f"Saved /tmp/{reg}-animation.gif")

animate(ntrials=10, duration=1000, reg='l2', dpi=120, contour_levels=100)
# animate(ntrials=10, duration=600, reg='l2', dpi=110, show_contours=False)