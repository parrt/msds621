import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # required even though not ref'd!
from matplotlib import rcParams
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_boston, load_iris, load_wine, load_digits, \
                             load_breast_cancer, load_diabetes, fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import glob
import os
from PIL import Image as PIL_Image

def loss(B,X,y):
    "Line coefficients: B = [y-intercept, slope]"
    return np.mean(y - np.dot(X,np.array(B)))**2

def get_surface(X, y, loss, b0_range, b1_range):
    n = len(X)
    B0 = np.ones(shape=(n, 1))
    X = np.hstack([np.ones(shape=(n, 1)), X]) # add ones column
    (b0_mesh, b1_mesh) = np.meshgrid(b0_range, b1_range, indexing='ij')
    L = np.zeros(b0_mesh.shape)

    for i in range(len(b0_range)):
        for j in range(len(b1_range)):
            L[i][j] = loss([b0_range[i],b1_range[j]], X=X, y=y)
    return L

def plot3d(L, b0_range, b1_range, ax, elev=50, azim=145):
    rcParams["font.size"] = 10
    ax.view_init(elev, azim)
    b0_range_mesh, b1_range_mesh = np.meshgrid(b0_range, b1_range, indexing='ij')
    surface = ax.plot_surface(b0_range_mesh, b1_range_mesh, L, alpha=0.7, cmap='coolwarm')
    
#    plt.title("""$loss(\\beta) = \sum_{i=1}^{N}(y^{{(i)}} - (\\beta_0 + \\beta_1 x^{{(i)}}))^2$""", fontsize=12)
    ax.set_xlabel('$\\beta_0$', fontsize=14)
    ax.set_ylabel('$\\beta_1$', fontsize=14)
    ax.zaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))

df = pd.read_csv('cheese_deaths.csv')

X, y = df.cheese.values, df.deaths.values
X = X.reshape(-1,1)
y = y.reshape(-1,1)

lm = LinearRegression()
lm.fit(X, y)
true_b0 = lm.intercept_[0]
true_b1 = lm.coef_[0][0]
print(f"True b0 = {lm.intercept_[0]:.2f}, b1 = {lm.coef_[0][0]:.1f}")

b0_range = np.arange(-3030, -2900, .1)  # y intercept
b1_range = np.arange(105, 120, .05)     # slope
L = get_surface(X, y, loss, b0_range=b0_range, b1_range=b1_range)

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')
plot3d(L, b0_range=b0_range, b1_range=b1_range, ax=ax, elev=25, azim=110)
ax.plot([true_b0], [true_b1], marker='x', markersize=10, color='black')
plt.show()
