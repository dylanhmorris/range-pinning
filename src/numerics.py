#!/usr/bin/env python3

# filename: numerics.py
# author: Dylan Morris <dhmorris@princeton.edu>
# description: numerics for Miller/Morris range
# pinning on a magic trait

import numpy as np
from scipy.integrate import simps
from scipy.stats import norm
import matplotlib.pyplot as plt

zmax = 4
xmax = 2
n_zs = 50
n_xs = 5
default_clutch_size = 1

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
        psi_x_zs,
        zs,
        xs,
        sigma_r,
        sigma_m,
        s,
        b):

    psi1, psi2, zvals = np.meshgrid(
        psi_x_zs,
        psi_x_zs,
        zs,
        sparse=True)
    
    print(psi1.shape, psi2.shape, zvals.shape)

    z1, z2 = np.meshgrid(zs, zs, sparse=True)
    integrand_grid = (
        psi1 * psi2 * P_o(zvals, z1, z2, sigma_r) *
        P_m(z1, z2, sigma_m))
    print('integrand grid shape:', integrand_grid.shape)
    integrated = (
        simps(simps(integrand_grid, zs), zs))

    result = fitness_w(zvals, s, xs, b) * integrated
    return result.flatten()


def psi_tilde_vec(
        psi_xs,
        zs,
        xs,
        sigma_r,
        sigma_m,
        s,
        b,
        clutch_size=default_clutch_size):

    psi1 = psi_xs[:, np.newaxis, :, np.newaxis]
    psi2 = psi_xs[:, :, np.newaxis, np.newaxis]
    zvals = zs[np.newaxis, np.newaxis, np.newaxis, :]
    
    z1, z2 = np.meshgrid(zs, zs, sparse=True)
    integrand_grid = (
        psi1 * psi2 * P_o(zvals, z1, z2, sigma_r) *
        P_m(z1, z2, sigma_m))

    integrated = (
        simps(simps(integrand_grid, zs), zs))

    return (clutch_size *
            fitness_w(zs[np.newaxis, :], s, xs[:, np.newaxis], b) *
            integrated)

def dispersal_kernel(x, y, sigma_d):
    return np.exp(-((x - y) ** 2) / (4 * (sigma_d ** 2)))


def psi_new(psi_old,
            zs,
            xs,
            sigma_r,
            sigma_m,
            s,
            b,
            sigma_d):
    psi_tilde = psi_tilde_vec(
        psi_old,
        zs,
        xs,
        sigma_r,
        sigma_m,
        s,
        b)

    x_vals = xs[np.newaxis, :]
    y_vals = xs[:, np.newaxis]

    kern = dispersal_kernel(x_vals, y_vals, sigma_d)
    normed_kern = (kern / np.sum(kern, axis=0)).transpose()
    ## everyone goes somewhere, dispersal probabilities
    ## row i, column j are prob disperse from x_i to x_j
    ## (so the rows sum to 1)

    integrand = normed_kern[:, :, np.newaxis] * psi_tilde[:, np.newaxis, :]
    ## with this broadcasting, integrand[i, j, k] is the
    ## number of individuals dispersing from x_i to x_j who
    ## are of trait z_k

    ## we now want a vector of the final total counts of z_k individuals
    ## at position x_j, so we want to sum over the first index x_i
    return simps(integrand, xs, axis=0)
    

zs = np.linspace(-zmax, zmax, n_zs)
xs = np.linspace(0, xmax, n_xs)

psis = np.vstack((
    [np.ones_like(zs)] +
    [np.zeros_like(zs)] * (len(xs) - 1)))


sigma_r = 10
sigma_m = 0.001
s = 0.001
b = 1
sigma_d = 0.02

a = psi_new(psis, zs, xs, sigma_r, sigma_m, s, b, sigma_d)
plt.plot(zs, a.transpose())
plt.legend(labels=xs)

for k in range(500):
    a = psi_new(a, zs, xs, sigma_r, sigma_m, s, b, sigma_d)
    a = a / np.sum(a)
plt.plot(zs, a.transpose())
