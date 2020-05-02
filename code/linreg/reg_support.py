import numpy as np
import math

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


def circle(lmbda=1, n=100):
    # walk radians around circle, using cos, sin to get x,y
    points = []
    for angle in np.linspace(0,np.pi/2, num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    for angle in np.linspace(np.pi/2,np.pi, num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    for angle in np.linspace(np.pi, np.pi*3/2, num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    for angle in np.linspace(np.pi*3/2, 2*np.pi,num=n//4):
        x = np.cos(angle) * lmbda
        y = np.sin(angle) * lmbda
        points.append((x,y))
    return np.array(points)


def loss(b0, b1,
         a = 1,
         b = 1,
         c = 0,     # axis stretch
         cx = -10,  # shift center x location
         cy = 5):   # shift center y
    return a * (b0 - cx) ** 2 + b * (b1 - cy) ** 2 + c * (b0 - cx) * (b1 - cy)


# def cirdist(a, b, r):
#     # compute the arctan of 'a' and 'b' then get angle between them
#     # then use ratio of full circle circumference to get distance
#     if np.isclose(a,0.0).any():
#         return
#     else:
#         theta = math.atan(b2/b1)
#     circum = 2*np.pi*r
#     frac_of_circum = circum * (theta / (2*np.pi))
#     return frac_of_circum

#print(circle(1,n=20))