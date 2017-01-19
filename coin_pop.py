##
## Estimating population of coin weights
##
## Yarden Katz <yarden@hms.harvard.edu>
##
import os
import sys
import time

import numpy as np
import scipy
import scipy.stats

import pymc3 as pm

import matplotlib.pylab as plt
import seaborn as sns

def make_data(num_coins, num_obs_per_coin,
              alpha_1, alpha_2):
    """
    Make data from population of coin weights.
    """
    coin_weights = np.random.beta(alpha_1, alpha_2, num_coins)
    data = []
    for n in xrange(num_coins):
        curr_weight = coin_weights[n]
        num_heads = np.random.binomial(num_obs_per_coin, curr_weight)
        num_tails = num_obs_per_coin - num_heads
        data.append([num_heads, num_tails])
    return np.array(data)


# make some data
num_coins = 10
num_obs_per_coin = 20
# true parameters of our Beta
#alpha_1, alpha_2 = 0.5, 0.5
alpha_1, alpha_2 = 5, 1
data = make_data(num_coins, num_obs_per_coin, alpha_1, alpha_2)
# summarize data as number of heads
data = data[:, 0]
print "Data: ",
print data

# inference parameters
num_samples = 1000

# need a prior on alpha_1, alpha_2
# parameterized by these hyperparameters
hyper_left, hyper_right = 1, 1

# our model assumes that the number of observations
# per coin is fixed; but this can be easily modeled
# as well
num_total_vector = np.ones(num_coins) * num_obs_per_coin
trace = None
with pm.Model() as peshkin:
    # Prior on parameters of Beta (which represents a *population*
    # of coin weights)
    alpha_left = pm.Beta("alpha_left", alpha=hyper_left, beta=hyper_right)
    alpha_right = pm.Beta("alpha_right", alpha=hyper_left, beta=hyper_right)
    # Now we're going to *vector of coin weights*, each
    # corresponding to our coin
    coin_weights = pm.Beta("coin_weights",
                           alpha=alpha_left, beta=alpha_right,
                           shape=num_coins)
    # Observations for each coin weight (drawn i.i.d*)
    data = pm.Binomial("obs", p=coin_weights, n=num_total_vector,
                       observed=data)
    t1 = time.time()
    step1 = pm.Metropolis([alpha_left, alpha_right, coin_weights])
    trace = pm.sample(num_samples, [step1])
    t2 = time.time()
    print "inference took %.2f mins" %((t2 - t1)/60.)
    print "trace: ", trace


print "ground truth: "
print "--"
print "alpha_1: %.3f" %(alpha_1)
print "alpha_2: %.3f" %(alpha_2)
print "\n"   
print "inferred results: "
print "---"
# take mean of posterior on parameters of Beta
mean_alpha_left = trace["alpha_left"].mean()
mean_alpha_right = trace["alpha_right"].mean()
print "alpha_left = %.3f" %(mean_alpha_left)
print "alpha_right = %.3f" %(mean_alpha_right)
print "inferred coin weights: "
inferred_weights = trace["coin_weights"].mean(axis=0)
    
# take the posterior mean of Beta's hyperparameters
# and plot them
plt.figure()
sns.set_style("ticks")
x = np.linspace(0., 1., 1000)
# ground truth Beta dist
plt.plot(x, scipy.stats.beta.pdf(x, alpha_1, alpha_2), label="True", color="k")
print mean_alpha_left
print mean_alpha_right
# inferred Beta dist
plt.plot(x, scipy.stats.beta.pdf(x, mean_alpha_left, mean_alpha_right),
         label="Inferred", color="r")
# plot each inferred coin weight
y_val = 2.
for weight in inferred_weights:
    plt.plot(weight, y_val, "x", marker="x", markersize=5, color="k")
plt.xlabel(r"$\theta$")
plt.legend()
plt.title("No. coins = %d, No. obs per coin = %d" %(num_coins,
                                                    num_obs_per_coin))
sns.despine(trim=True)
plt.savefig("simulation_results.pdf")
    
    

    
