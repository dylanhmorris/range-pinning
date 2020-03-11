################################################
# filename: numerics.py
# author: Dylan Morris <dhmorris@princeton.edu>
# description: numerics for Miller/Morris range
# pinning on a magic trait
################################################

import numpy as np
from scipy.integrate import simps
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

def P_m(z_1, z_2, sigma_m, all_zs):
    norm_const = np.sum([np.exp(-((z_1 - z_j) ** 2) / (sigma_m ** 2))
                         for z_j in all_zs])
    return np.exp(-((z_1 - z_2) ** 2) / (sigma_m ** 2)) / norm_const

def P_o(z, z_1, z_2, sigma_r):
    """
    returns a tensor such that entry 
    i, j, k is the probability that an
    individual of type z_i is produced
    by a mating between a 1st individual
    of type z_j and a second individual of
    type z_k
    """
    ## do some outer sums and division to get our n x n x n tensor
    z_mid = (z_1[:, np.newaxis] + z_2[np.newaxis, :]) / 2
    unnormalized_probs = np.exp(-((z[:,np.newaxis, np.newaxis] - z_mid[np.newaxis, :, :]) ** 2)
                                / (sigma_r ** 2))
    norm_mat = np.sum(unnormalized_probs, axis=0)
    
    ## division is along trailing axes, so this works
    ## without any extra broadcasting tricks
    return unnormalized_probs / norm_mat 


def P_o_old(z, z_1, z_2, sigma_r, all_zs):
    z_mid = (z_1 + z_2) / 2
    unnormalized_probs = np.exp(-((z - z_mid) ** 2)
                                / (sigma_r ** 2))
    norm_const = np.sum([np.exp(-((z_j - z_mid) ** 2) / (sigma_r ** 2))
                         for z_j in all_zs])
    return unnormalized_probs / norm_const

def P_o_vec(zs, z_1s, z_2s, sigma_r, all_zs):
    return np.array(
        [P_o_old(z, z_1, z_2, sigma_r, all_zs) for z in zs
         for z_1 in z_1s
         for z_2 in z_2s]).reshape((zs.size, z_1s.size, z_2s.size))

def z_opt(x, b):
    return b * x

def fitness_w(z, s, x, b):
    return np.exp(-s * ((z - z_opt(x, b)) ** 2))


def dispersal_kernel(x, y, sigma_d):
    return np.exp(-((x - y) ** 2) / (4 * (sigma_d ** 2)))


class RangePinningModel():

    def __init__(self,
                 zmax,
                 xmax,
                 n_zs,
                 n_xs,
                 clutch_size,
                 carrying_capacity,
                 sigma_r,
                 sigma_m,
                 sigma_d,
                 s,
                 b,
                 continuous_space=False):

        self.zmax = zmax
        self.xmax = xmax
        self.n_zs = n_zs
        self.n_xs = n_xs
        self.clutch_size = clutch_size
        self.carrying_capacity = carrying_capacity        
        self.sigma_r = sigma_r    # trait sd (in recombination)

        self.sigma_m = sigma_m
        # 1 / assortativity of mating
        # (sd of assortativity kernel)
                                  
        self.sigma_d = sigma_d    # dispersal kernel sd
        self.s = s                # strength of selection
        self.b = b  # environmental trait optimum gradient steepness

        self.continuous_space = continuous_space

        self.reset()

        return None


    ## normalize populations by fineness
    ## of space
    def normed_clutch_size(self):
        return self.clutch_size / self.n_xs

    def normed_carrying_capacity(self):
        return self.carrying_capacity / self.n_xs    
    
    def psi_tilde_vec(
            self,
            verbose=False):

        psi1 = self.psi_vec[:, np.newaxis, :]
        psi2 = self.psi_vec[:, :, np.newaxis]
    
        z1, z2 = np.meshgrid(self.zs, self.zs, sparse=True)

        n_encounters = psi1 * psi2
        ## this might be a problem. need to think what assortativity
        ## should look like. Here we maybe have a problem because
        ## you always generate more individuals if you always mate
        prob_mate_given_encounter = P_m(z1, z2, self.sigma_m, self.zs)
    
        n_mating_encounters = (n_encounters *
                               prob_mate_given_encounter[np.newaxis, :, :])

        offspring_by_parent_pair = (
            self.prob_phenotype_given_parents[np.newaxis, :, :, :] *
            n_mating_encounters[:, np.newaxis, :, :])

        ## n_xs by n_zs by n_zs by n_zs tensor
        ## i, j, k, l^th entry gives offspring in
        ## the i^th deme of type z_j resulting
        ## from matings between a z_k first parent
        ## and a z_l second parent

        ## now we integrate that over the last two
        ## axes to obtain an n_xs x n_zs matrix
        ## of offspring by deme
        integrated = (
            np.sum(
                np.sum(offspring_by_parent_pair,
                       axis = -1),
                axis = -1))

        if verbose:
            print("mating probs given encounter: \n", prob_mate_given_encounter)
            print("encounters: \n", n_encounters)
            print("mating encounters: \n", n_mating_encounters)
            print("phenotype given parents: \n", self.prob_phenotype_given_parents)
            print("offspring by parent pair: \n", offspring_by_parent_pair)
            print("integrated: \n", integrated)
            print("mating probs shape:", prob_mate_given_encounter.shape)
            print("mating encounters shape:", n_mating_encounters.shape)
            print("phenotype given parents shape:", self.prob_phenotype_given_parents.shape)
            print("offspring by parent pair shape:", offspring_by_parent_pair.shape)
            print("integrated shape: ", integrated.shape)

        return (self.normed_clutch_size() *
                fitness_w(self.zs[np.newaxis, :], self.s, self.xs[:, np.newaxis], self.b) *
                integrated)

    def psi_new(
            self,
            verbose=False):
    
        psi_tilde = self.psi_tilde_vec(
            verbose=verbose)

        if verbose:
            print(np.sum(psi_tilde))
    
        x_vals = self.xs[np.newaxis, :]
        y_vals = self.xs[:, np.newaxis]

        kern = dispersal_kernel(x_vals, y_vals, self.sigma_d)
        normed_kern = (kern / np.sum(kern, axis=0)).transpose()
        ## everyone goes somewhere, dispersal probabilities
        ## row i, column j are prob disperse from x_i to x_j
        ## (so the rows sum to 1)
        
        ## with this broadcasting, integrand[i, j, k] is the
        ## number of individuals dispersing from x_i to x_j who
        ## are of trait z_k
        integrand = normed_kern[:, :, np.newaxis] * psi_tilde[:, np.newaxis, :]

        if verbose:
            print("integration check:", np.sum(psi_tilde), np.sum(integrand))
            

        ## we now want a vector of the final total counts of z_k individuals
        ## at position x_j, so we want to sum over the first index x_i
        if self.continuous_space:
            vec = simps(integrand, xs, axis=0)
        else:
            vec = np.sum(integrand, axis=0)

        total_pop_in_space = np.sum(vec, axis = 1)
        pops_vec_shape = total_pop_in_space[:, np.newaxis] * np.ones_like(vec)
        scaling_term = self.normed_carrying_capacity() / np.maximum(pops_vec_shape, 1e-6)
        vec = vec * ((pops_vec_shape > self.carrying_capacity) * scaling_term +
                     (pops_vec_shape <= self.carrying_capacity) * 1)
        return vec

    def reset(self):
        if self.zmax < 2 * self.b * self.xmax:
            warnings.warn("Small zmax relative to "
                          "xmax. May pin due to unavailable "
                          "phenotypes preventing adaptation.")

        self.zs = np.linspace(-self.zmax, self.zmax, self.n_zs)
        self.xs = np.linspace(-self.xmax, self.xmax, self.n_xs)

        self.psi_vec = np.vstack(
            [np.zeros_like(self.zs)] * len(self.xs)
        )

        # initialize population at the center with individuals
        # of all phenotypes
        self.psi_vec[int(self.n_xs/2), int(self.n_zs/2)] = self.carrying_capacity
        self.calc_P_o()
        
    def update(self, niter,
               verbose = False,
               print_iter = 100):
        ## only calculate P_o once, as it is constant given
        ## the zs and parameters
        self.calc_P_o()

        for k in range(niter):
            
            self.psi_vec = self.psi_new(
                verbose=verbose)
            
            if k % print_iter == 0:
                print("Iteration no. {}".format(k))

            pass
        
        return 0

    def calc_P_o(self):
        self.prob_phenotype_given_parents = P_o(self.zs, self.zs, self.zs, self.sigma_r)

    def update_fig(self):
            
        def label(x, color, label):
            ax = plt.gca()
            variable_val = x.iloc[0]
            ax.text(0, .2, '{0:.2f}'.format(variable_val),
                    fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)
            return 0
            
        adf = pd.DataFrame(np.vstack([self.zs, self.psi_vec]).transpose(),
                           columns=['z_value'] + [x for x in self.xs])

        adf = adf.melt(id_vars=['z_value'])

        g = sns.FacetGrid(adf, row="variable", hue="variable",
                          aspect=15, height = 0.5)
        
        g.map(label, "variable")
        g.map(plt.plot, 'z_value', 'value')
        g.fig.subplots_adjust(hspace=-.05)
        g.set_titles("")
        g.despine(bottom=True,
                  left=True)

        return g

    def total_pop(self):
        return np.sum(self.psi_vec)

    def _which_x(self, x):
        return np.where(model.xs >= 5)[0][0]

    def pop_at_xs(self, x_val):
        return self.psi_vec[self._which_x(x_val)]

    
########
# these parameters range pin with the
# larger sigma_m but not the smaller
#########

zmax_init = 10
xmax_init = 5
n_zs_init = 50
n_xs_init = 15

clutch_size_init = 10000
carrying_capacity_init = 100

sigma_r_init = 0.1    # trait sd (in recombination)
sigma_m_init = 15     # assortativity of mating sd
s_init = 2            # strength of selection
b_init = 1            # environmental trait optimum gradient steepness
sigma_d_init = 0.25   # dispersal kernel sd
##########################################

model = RangePinningModel(
    zmax_init,
    xmax_init,
    n_zs_init,
    n_xs_init,
    clutch_size_init,
    carrying_capacity_init,
    sigma_r_init,
    sigma_m_init,
    sigma_d_init,
    s_init,
    b_init)
    

print(model.total_pop())


