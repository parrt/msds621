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

w,h = 10,10
beta0 = np.linspace(-w, w, 100)
beta1 = np.linspace(-h, h, 100)
B0, B1 = np.meshgrid(beta0, beta1)

def loss(b0, b1,
         a = 1,
         b = 1,
         c = 0,     # axis stretch
         cx = -10,  # shift center x location
         cy = 5):   # shift center y
    return a * (b0 - cx) ** 2 + b * (b1 - cy) ** 2 + c * (b0 - cx) * (b1 - cy)


def select_parameters():
    while True:
        a = np.random.random() * 10
        b = np.random.random() * 10
        c = np.random.random() * 4 - 1.5
        x = 0
        while np.abs(x) < 1:
            x = np.random.random() * 2 * w - w
        y = 0
        while np.abs(y) < 1:
            y = np.random.random() * 2 * h - h

        Z = loss(B0, B1, a=a, b=b, c=c, cx=x, cy=y)
        loss_at_min = loss(x, y, a=a, b=b, c=c, cx=x, cy=y)
        if (Z >= loss_at_min).all(): # hooray! we didn't make a saddle point
            break # fake repeat-until in python
        print("loss not min", loss_at_min)

    return Z, a, b, c, x, y


def plot_loss(boundary, show_contours=True, contour_levels=50, show_loss_eqn=False):
    Z, a, b, c, x, y = select_parameters()
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


def animate(ntrials=20, dpi=100, duration=600):
    plt.close()
    for f in glob.glob(f'/tmp/L1-frame-*.png'):
        os.remove(f)

    dia = diamond(lmbda=2, n=100)
    for i in range(ntrials):
        plot_loss(boundary=dia)
        print(f"/tmp/frame-{i}.png")
        plt.savefig(f"/tmp/frame-{i}.png", bbox_inches=0, pad_inches=0, dpi=dpi)
        plt.close()

    images = [PIL_Image.open(image) for image in sorted(glob.glob(f'/tmp/frame-*.png'))]
    images += reversed(images)
    images[0].save(f'/tmp/L1-animation.gif',
                   save_all=True,
                   append_images=images[1:],
                   duration=duration,
                   optimize=False,
                   loop=0)
    print("Saved /tmp/L1-animation.gif")

np.random.seed(1)

animate(duration=600)