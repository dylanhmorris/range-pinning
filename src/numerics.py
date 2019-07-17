#!/usr/bin/env python3

# filename: numerics.py
# author: Dylan Morris <dhmorris@princeton.edu>
# description: numerics for Miller/Morris range
# pinning on a magic trait

import numpy as np
from scipy.integrate import simps
from scipy.stats import norm
import time

zmax = 4
n_zs = 100

def P_m(z_1, z_2, sigma_m):
    return np.exp(-((z_1 - z_2) ** 2) / (sigma_m ** 2))

def P_o(z, z_1, z_2, sigma_r):
    z_mid = (z_1 + z_2) / 2
    return np.exp(-((z - z_mid) ** 2) / (sigma_r ** 2))

def z_opt(x, b):
    return b * x

def fitness_w(z, s, x, b):
    return np.exp(-s * ((z - z_opt(x, b)) ** 2))

def psi_tilde(
        zs,
        x,
        psi_x_zs,
        sigma_r,
        sigma_m,
        s,
        b):
    
    psi1, psi2, zvals= np.meshgrid(
        psi_x_zs,
        psi_x_zs,
        zs,
        sparse=True)
    
    z1, z2 = np.meshgrid(zs, zs, sparse=True)
    integrand_grid = (
        psi1 * psi2 * P_o(zvals, z1, z2, sigma_r) *
        P_m(z1, z2, sigma_m))

    integrated = (
        simps(simps(integrand_grid, zs), zs))

    new_psi = fitness_w(zvals, s, x, b) * integrated
    
    return (new_psi / np.sum(new_psi)).flatten()

    
def dispersal_kernel(x, y, sigma_d):
    return np.exp(-(x - y) ** 2 / (4 * sigma_d ** 2))
    
    

plt.plot(zs, psi_x_zs)
a = psi_tilde(zs, 1, psi_x_zs, 1, 1, 0.5, 0.5)
plt.plot(zs, a)

for k in range(10):
    a = psi_tilde(zs, 1, a, 1, 1, 0.5, 0.5)
    plt.plot(zs, a)
