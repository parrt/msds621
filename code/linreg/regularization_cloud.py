import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image as PIL_Image
from reg_support import *
from scipy.spatial.distance import euclidean
from colour import Color

GREY = '#444443'

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


def plot_cloud(lmbda, reg, n_trials, zero_color = '#40DE2D',
               nonzero_color = '#3C659D',
               nonzero_color_l2 = '#F46C43',
               ncolors=100, dpi=200):
    zeroes = [(0,lmbda), (lmbda,0), (0,-lmbda), (-lmbda,0)]
    if reg=='l1':
        boundary = diamond(lmbda=lmbda)
    else:
        boundary = circle(lmbda=lmbda)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks([-10, -5, 0, 5, 10])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.margins(0)

    # Draw axes
    ax.plot([-w, +w], [0, 0], '-', c='k', lw=.8)
    ax.plot([0, 0], [-h, h], '-', c='k', lw=.8)

    # Draw boundary
    ax.plot(boundary[:, 0], boundary[:, 1], '-', lw=.8, c='#A22396')

    c = 0 # count
    centers = []
    centers0 = []
    for i in range(n_trials):
        print(i)
        Z, a, b, color, x, y = select_parameters(lmbda, reg)

        # Find point on boundary
        losses = [loss(*edgeloc, a=a, b=b, c=color, cx=x, cy=y) for edgeloc in boundary]
        minloss_idx = np.argmin(losses)
        coeff = boundary[minloss_idx]
        if np.isclose(coeff, 0).any():
            c += 1
            centers0.append((x,y))
        else:
            centers.append((x,y))

    if reg=='l2':
        # what is distance from x,y to a zero?
        distances_to_0 = []
        for x,y in centers:
            d0 = [euclidean((x, y), z) for z in zeroes]
            distances_to_0.append( np.min(d0) )
        distances_to_0 = np.array(distances_to_0)

        # use order of distance to color so must be very close to 0 to be green
        # distances_to_0 = np.log(distances_to_0)
        distances_to_0 = np.sqrt(distances_to_0)*3
        dmin = np.min(distances_to_0)
        dmax = np.max(distances_to_0)
        drange = dmax - dmin

        normalized_distances_to_0 = (distances_to_0-dmin)/drange

        blue = Color(nonzero_color_l2)
        green = Color(zero_color)
        spectrum = np.array(list(green.range_to(blue, ncolors)))

        colors = spectrum[(normalized_distances_to_0*(ncolors-1)).astype(int)]
        colors = [c.rgb for c in colors]
    else:
        colors = nonzero_color

    # Draw centers of loss functions
    centers = np.array(centers)
    centers0 = np.array(centers0)
    minority_alpha = 1
    majority_alpha = .5
    if reg=='l1':
        a = minority_alpha
        a0 = majority_alpha
    else:
        a = majority_alpha
        a0 = minority_alpha
    ax.scatter(centers[:,0], centers[:,1], s=20, c=colors, alpha=a)
    ax.scatter(centers0[:,0], centers0[:,1], s=20, c=zero_color, alpha=a0)

    ax.set_title(f"Loss function minimum cloud\n{reg.upper()} gives {100*c//n_trials}% zeroes", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"/tmp/{reg}-cloud.png", bbox_inches=0, pad_inches=0, dpi=dpi)
    plt.show()


plot_cloud(lmbda=2, reg='l1', n_trials=10000, ncolors=100)