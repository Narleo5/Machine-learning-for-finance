import sys
import os
sys.path.append(os.path.dirname('__file__'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time
from random import randrange
import copy
import argparse
import random
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

    
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

## Load market prices and set training target

## data from the article
data = torch.load("Call_prices_59.pt")
# data = data[:6,:13]

# Set up training - Strike values, time discretisation and maturities
strikes_call = np.linspace(0.8, 1.2, 21)
n_steps = 96
maturities = [60, 120, 180, 240, 300, 360]
maturities = [m // 30 for m in maturities]
n_maturities = len(maturities)

## Neural SDE
S0 = 1
rate = 0.025 # risk-free rate

titles = ['price_vanilla_cv', 'var_price_vanilla_cv']
log_moneyness = np.array([np.log(K/S0) for K in strikes_call])

## plot target data
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = log_moneyness, np.array(maturities) # ou strikes_call, np.array(maturities)
X, Y = np.meshgrid(X, Y)

mycmap = plt.get_cmap('jet')
surf = ax.plot_surface(X, Y, data, cmap=mycmap)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('log-moneyness')
ax.set_ylabel('maturity (1=30d)')
ax.set_zlabel('target data')

plt.show()

###
#df = pd.DataFrame(data, index=[m/12 for m in maturities], columns=strikes_call)

from scipy.stats import norm
from scipy.optimize import bisect

def BS_original(s, k, r, sigma, T):
    d1 = (np.log(s/k) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = s * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    return C

def vol_diff(mkt, s, k, r, sigma, T):
    return mkt - BS_original(s, k, r, sigma, T)

def get_iv(mkt, s, k, r, T):
    def vol_diff_wrapped(sigma):
        return vol_diff(mkt, s, k, r, sigma, T)
    return bisect(vol_diff_wrapped, 1e-9, 10)

iv_mtx = []
mat_norm = [m / 12 for m in maturities]
for m in range(len(maturities)):
    line = []
    for k in range(len(strikes_call)):
        iv = get_iv(data[m, k], S0, strikes_call[k], rate, mat_norm[m])
        line.append(iv)
    iv_mtx.append(line)
iv_mtx = np.array(iv_mtx)

## plot iv
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

X, Y = log_moneyness, np.array(mat_norm) # ou strikes_call, np.array(maturities)
X, Y = np.meshgrid(X, Y)

mycmap = plt.get_cmap('jet')
surf = ax.plot_surface(X, Y, iv_mtx, cmap=mycmap)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel('log-moneyness')
ax.set_ylabel('maturity')
ax.set_zlabel('target data')

plt.show()

