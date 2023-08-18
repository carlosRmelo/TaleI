import os
import sys
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares
from scipy import interpolate
import pickle
from scipy.stats.mstats import mquantiles
import json

"""
The ideia is to derive the total mass density slope.
To that we are using two approachs.
 1. Fitting a power-law profile in a log-log space (Li+16 DOI:10.1093/mnras/stv2565)
 2. Computing the average slope (Xu+2017 DOI:10.1093/mnras/stx899)

Here we derive the slope from the model, using the Distributed mass profile.
"""

def PL_model(theta, x):
    """
        This is a power law model in a log-log space.
        Inputs:
        ---------------
        theta[0]: the slope
        theta[1]: the normalisation
        x       : the logarithmic distance
    """
    return theta[0]*x + theta[1]

def fit_funtion(theta, log_rho, log_r):
    """
        Fit function for a least_squares.
        Inputs:
        ---------------
        theta[0]: the slope
        theta[1]: the normalisation
        log_rho : logarithmic density (data)
        log_r   : logarithmic distance (data)
        
    """
    return PL_model(theta, log_r) - log_rho

def restrict_density(density, r, r1, r2):
    """
        Return the density and the radius between r1 and r2.
        r1 < r2
    """
    assert r1 < r2, "r2 should be greater than r1."
    
    i = (r1 < r) &  (r < r2)
    
    return density[i], r[i]

def AV_slope(density, r, r1, r2):
    """
        Compute the average density slope as defined by
        eq. (15) in Xu+2017.
        The density profile is interpolated at r1 and r2
        using scipy.interpolate.interp1d. We use a cubic 
        interpolation.
        
        Inputs:
        ---------------
        density: density profile covering the range of r1 and r2
        r: radius where the density is evaluated
        r1, r2: inner and outter radius where the slope will be evaluated
    """
    
    f_ = interpolate.interp1d(r, density, kind="cubic")
    return np.log( f_(r2) / f_(r1) ) / np.log( r1 / r2)
    
def run(result_path):
    with open(result_path+f'/densities/stellar_density3D_distribution.pickle','rb') as f:
        stellar_density_dist = pickle.load(f)
        f.close()
            
    with open(result_path+f'/densities/dm_density3D_distribution.pickle','rb') as f:
        dm_density_dist = pickle.load(f)
        f.close()

    with open(result_path+"/quantities.json") as f:
            quantities = json.load(f)
            f.close()
            
    with open(result_path+"/description.json") as f:
            description = json.load(f)
            f.close()

    reff = quantities["Reff"]
    # Load DM density profile
    rho_dm = mquantiles(dm_density_dist["distribution"], 0.5, axis=0)[0]
    r_dm   = dm_density_dist["radii"]

    # Load star density profile
    rho_star = mquantiles(stellar_density_dist["distribution"], 0.5, axis=0)[0]
    r_star   = stellar_density_dist["radii"]

    # Total mass density profile
    rho_total = rho_dm + rho_star
    r_total   = r_dm

    description["PL_slope1"] = "PL slope within Rmin-1.5Reff"
    description["PL_slope2"] = "PL slope within 1.5Reff-2.5Reff"
    description["AV_slope1"] = "average slope within Rmin-1.5Reff"
    description["AV_slope2"] = "average slope within 1.5Reff-2.5Reff"

    out_descripition = open("{}/description.json".format(result_path), "w")
    json.dump(description, out_descripition, indent = 8)
    out_descripition.close()

    theta0 = [-2, 5]   # Initial guess. [slope, normalisation]
    r1 = r_total.min()
    r2 = 2 * reff
    rho, r = restrict_density(density=rho_total, r=r_total, r1=r1, r2=r2)         # Get density within r1 and r2

    fit_total = least_squares(fit_funtion, theta0, args=(np.log(rho), np.log(r))) # Fit
    PL_slope1 = fit_total.x[0]
    AV_slope1 = AV_slope(rho_total, r_total, r1, r2)  # Get the average slope
    quantities["PL_slope1"] = PL_slope1
    quantities["AV_slope1"] = AV_slope1

    theta0 = [-2, 5]   # Initial guess. [slope, normalisation]
    r1 = 2 * reff
    r2 = 2.5 * reff
    rho, r = restrict_density(density=rho_total, r=r_total, r1=r1, r2=r2)         # Get density within r1 and r2

    fit_total = least_squares(fit_funtion, theta0, args=(np.log(rho), np.log(r))) # Fit
    PL_slope2 = fit_total.x[0]
    AV_slope2 = AV_slope(rho_total, r_total, r1, r2)  # Get the average slope
    quantities["PL_slope2"] = PL_slope2
    quantities["AV_slope2"] = AV_slope2

    out_quantiles = open("{}/quantities.json".format(result_path), "w")
    json.dump(quantities, out_quantiles, indent = 8)
    out_quantiles.close()


if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print('Error - please provide a folder name')
        sys.exit(1)
    path = args[0]
    run(path)
    sys.exit()
