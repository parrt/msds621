import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image as PIL_Image
from reg_support import *
from scipy.spatial.distance import euclidean
from colour import Color
import math

GREY = '#444443'

w,h = 10,10
beta0 = np.linspace(-w, w, 100)
beta1 = np.linspace(-h, h, 100)
B0, B1 = np.meshgrid(beta0, beta1)


def select_parameters(lmbda, reg, force_symmetric_loss, force_one_nonpredictive):
    while True:
        a = np.random.random() * 10
        b = np.random.random() * 10
        c = np.random.random() * 4 - 2
        if force_symmetric_loss:
            b = a # make symmetric
            c = 0
        elif force_one_nonpredictive:
            if np.random.random() > 0.5:
                a = np.random.random() * 15 - 5
                b = .1
            else:
                b = np.random.random() * 15 - 5
                a = .1
            c = 0

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

    eqn = f"{a:.2f}(b0 - {x:.2f})^2 + {b:.2f}(b1 - {y:.2f})^2 + {c:.2f} (b0-{x:.2f}) (b1-{y:.2f})"
    # print(eqn)
    return Z, a, b, c, x, y


def plot_cloud(lmbda, reg, n_trials,
               force_symmetric_loss=False, force_one_nonpredictive=False,
               zero_color = '#40DE2D',  # zero_edgecolor='#40DE2D',
               nonzero_color = '#8073ac',
               nonzero_color_l2 = '#8073ac',
               close_to_zero_color_l2 ='#A3E78B',
               isclose_threshold=0.1,
               markersize = 3,
               ncolors=100, dpi=200):
    zeroes = [(0,lmbda), (lmbda,0), (0,-lmbda), (-lmbda,0)]
    if reg=='l1':
        boundary = diamond(lmbda=lmbda)
    else:
        boundary = circle(lmbda=lmbda)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
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
    coeffs = []
    centers = []
    centers0 = []
    for i in range(n_trials):
        Z, a, b, color, x, y = select_parameters(lmbda, reg,
                                                 force_symmetric_loss=force_symmetric_loss,
                                                 force_one_nonpredictive=force_one_nonpredictive)

        # Find point on boundary
        losses = [loss(*edgeloc, a=a, b=b, c=color, cx=x, cy=y) for edgeloc in boundary]
        minloss_idx = np.argmin(losses)
        coeff = boundary[minloss_idx]
        if np.isclose(coeff, 0).any():
            c += 1
            centers0.append((x,y))
        else:
            coeffs.append(coeff)
            centers.append((x,y))

    if reg=='l2':
        # what is Euclidean distance from beta1,beta2 to a zero?
        # should use distance around circle but this works for visualization purposes
        distances_to_0 = []
        for x,y in coeffs:
            d0 = [euclidean((x, y), z) for z in zeroes]
            distances_to_0.append( np.min(d0) )
        distances_to_0 = np.array(distances_to_0)

        # use order of distance to color so must be very close to 0 to be green
        # distances_to_0 = np.log(distances_to_0)
        distances_to_0 = np.sqrt(distances_to_0)*3
        dmin = np.min(distances_to_0)
        dmax = np.max(distances_to_0)
        drange = dmax - dmin

        # print("DIST", dmin, dmax)

        normalized_distances_to_0 = (distances_to_0-dmin)/drange
        # thresholded_distances_to_0 = np.clip(normalized_distances_to_0, 0.0, isclose_threshold)
        thresholded_distances_to_0 = np.where(normalized_distances_to_0 < isclose_threshold,
                                              normalized_distances_to_0, 1.0)

        nonzero_color_C = Color(close_to_zero_color_l2)
        zero_color_C = Color(zero_color)
        spectrum = np.array(list(zero_color_C.range_to(nonzero_color_C, ncolors)))

        # threshold_0_to_1 = normalized_distances_to_0 * (1 / isclose_threshold)
        # colors = spectrum[(threshold_0_to_1 * (ncolors - 1)).astype(int)]
        isclose_idx = np.where(normalized_distances_to_0 < isclose_threshold)[0]
        close_colors = spectrum[(normalized_distances_to_0[isclose_idx] * (1/isclose_threshold) * (ncolors - 1)).astype(int)]
        colors = np.full(shape=(len(distances_to_0)), dtype=object, fill_value=Color(nonzero_color_l2))
        colors[isclose_idx] = close_colors
        colors[isclose_idx] = Color(close_to_zero_color_l2)
        colors = [c.rgb for c in colors]
    else:
        colors = nonzero_color

    # Draw centers of loss functions
    centers = np.array(centers)
    centers0 = np.array(centers0)
    if reg == 'l1' and force_symmetric_loss:
        ax.scatter(centers0[:, 0], centers0[:, 1], s=markersize, c=zero_color, alpha=1.0)
        ax.scatter(centers[:, 0], centers[:, 1], s=markersize+2, c=colors, alpha=1.0)
    elif reg=='l1' and force_one_nonpredictive:
        ax.scatter(centers[:, 0], centers[:, 1], s=markersize+5, c=colors, alpha=1.0)
        ax.scatter(centers0[:, 0], centers0[:, 1], s=markersize, c=zero_color, alpha=1.0)
    elif reg == 'l1':
        ax.scatter(centers[:, 0], centers[:, 1], s=markersize+5, c=colors, alpha=1.0)
        ax.scatter(centers0[:, 0], centers0[:, 1], s=markersize, c=zero_color, alpha=1.0)
    elif reg == 'l2' and force_one_nonpredictive:
        ax.scatter(centers[:, 0], centers[:, 1], s=markersize, c=colors, alpha=1.0)
        ax.scatter(centers0[:, 0], centers0[:, 1], s=markersize, c=zero_color, alpha=1.0)
    elif reg == 'l2':
        ax.scatter(centers[:, 0],  centers[:, 1], s=markersize, c=colors, alpha=1.0)
        ax.scatter(centers0[:, 0], centers0[:, 1], s=markersize+8, c=zero_color, alpha=1.0)

    shape = shape_fname = ""
    if force_symmetric_loss:
        shape_fname = "symmetric-"
        shape = "Symmetric "
    elif force_one_nonpredictive:
        shape_fname = "orthogonal-"
        shape = "Orthogonal "

    ax.set_title(f"{shape}Loss function min cloud\n{reg.upper()} gives {100*c//n_trials}% zeroes", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"/tmp/{reg}-{shape_fname}cloud.png", bbox_inches=0, pad_inches=0, dpi=dpi)
    plt.show()

sage = '#D7E2A9'
faint_blue = '#E4EDF1'
darker_faint_blue_edge = '#4096B6'
light_blue_metal = '#A4B7C7'
dark_mustard = '#E1BD4D'

n = 5000
# plot_cloud(lmbda=2, reg='l1', n_trials=n, ncolors=100,
#            zero_color='#21FF36', nonzero_color='#59A3D2')
# plot_cloud(lmbda=2, reg='l1', n_trials=n, ncolors=100, force_symmetric_loss=True,
#            zero_color='#21FF36', nonzero_color='#59A3D2')
# plot_cloud(lmbda=2, reg='l1', n_trials=n, ncolors=100, force_one_nonpredictive=True,
#            zero_color='#21FF36', nonzero_color='#59A3D2')

plot_cloud(lmbda=2, reg='l2', n_trials=n, ncolors=100,
           zero_color='#21FF36',
           close_to_zero_color_l2='#FEE08F',
           nonzero_color_l2='#59A3D2',
           isclose_threshold=0.3)
plot_cloud(lmbda=2, reg='l2', n_trials=n, ncolors=100, force_symmetric_loss=True,
           zero_color='#21FF36',
           close_to_zero_color_l2='#FEE08F',
           nonzero_color_l2='#59A3D2',
           isclose_threshold=0.3)
plot_cloud(lmbda=2, reg='l2', n_trials=n, ncolors=100, force_one_nonpredictive=True,
           zero_color='#21FF36',
           close_to_zero_color_l2='#FEE08F',
           nonzero_color_l2='#59A3D2',
           isclose_threshold=0.3)
