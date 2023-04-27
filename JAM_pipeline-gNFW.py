# ## Testing the a full Pipeline
# 
# Model: Analytical Dark matter component, black hole, scalar anisotropy, 
# scalar mass-to-light ratio. The ideia is testing the TNG50 simulation
# 
# Date: 16/03/2023
# 

import autolens as al
import autolens.plot as aplt

import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt

from dyLens import Lens
from dyLens.Dynamics import JAM
from dyLens.utils import priors as p
from dyLens.utils import dark_mge
from dyLens.Combined import CombinedModel



from os import path
from optparse import OptionParser

def dist_circle(xc, yc, s):
    """
    Returns an array in which the value of each element is its distance from
    a specified center. Useful for masking inside a circular aperture.

    The (xc, yc) coordinates are the ones one can read on the figure axes
    e.g. when plotting the result of my find_galaxy() procedure.

    """
    x, y = np.ogrid[-yc:s[0] - yc, -xc:s[1] - xc]   # note yc before xc
    rad = np.sqrt(x**2 + y**2)

    return rad


def run(data_path):
        #Reading inputs
    log = fits.open('{}/log_img.fits'.format(data_path))[1].data

        #IFU data
    hdulist = fits.open('%s/IFU.fits'%data_path)[1]
    tem = hdulist.data
    x0 = tem['xbin']
    y0 = tem['ybin']
    v0 = tem['v0']
    v0_err = tem['v0_err']
    vd = tem['vd']
    vd_err = tem['vd_err']
    vrms = tem['vrms']
    vrms_err = tem['vrms_err']
    r  = np.sqrt(x0**2+y0**2)
    ii = np.where( (0.01 < r) & (r < 6.0))

    pixsize = 0.2


    x0 = x0[ii]
    y0 = y0[ii]
    vrms = vrms[ii]
    vrms_err = vrms_err[ii]
        #MGE components
    sol_star, pa, eps, xmed, ymed = np.load('%s/mge.npy'%data_path,  allow_pickle=True)

        #cosmology info
    info  = fits.open('{}/image.fits'.format(data_path))[0].header
    cosmo = FlatLambdaCDM(H0=info["H0"], Om0=info["OMEGA0"])

        #Combined model building
    z_l     = info["z_l"]                    #lens Redshift

    surf  = sol_star[:,0]
    sigma = sol_star[:,1]
    qObs  = sol_star[:,2]


    # Initialize the class
    Jam_Model = JAM.axi_rms(ybin=y0, xbin=x0, z=z_l, rms=vrms, erms=0.11*vrms,
                            tensor="zz", 
                            pixsize=pixsize,
                            quiet=True)

    # Define the MGE inputs
    Jam_Model.components(surf_lum=surf, sigma_lum=sigma, qobs_lum=qObs, )
    # Define the model
    Jam_Model.set_anisotropy(beta_kind="scalar") #Seting vector anisotropy
    Jam_Model.set_mass_to_light(ml_kind="scalar")
    Jam_Model.include_BH()
    Jam_Model.include_dm("gNFW")


    #Build the model, and passing the priors
    parsDic = {
                "inc":     p.UniformPrior(40, 90.0),
                "beta":    p.UniformPrior(-0.5, 0.5),
                "log_mbh": p.UniformPrior(7.5, 9.5),
                "ml":      p.UniformPrior(1.0, 10),
                "log_rho_s":  p.UniformPrior(-6, 0),
                "rs":      p.UniformPrior(1e-2, 30),
                "qDM":     1.0,
                "slope":   p.UniformPrior(0.5, 2.0),
                }


    Jam_Model.build_model(parsDic)

    #Config non-linear search
    ncores = 5
    Jam_Model.config_non_linear(nlive=500, n_cores=ncores,)
    output_path = data_path.split("/")[1]+"/model2"

    Jam_Model.run_dynesty(maxiter=20, output_path=output_path)


if __name__ == '__main__':
    parser = OptionParser() #You also should inform the folder name

    (options, args) = parser.parse_args()
    if len(args) != 1:
        print('Error - please provide the path to the date.')
        sys.exit(1)
    data_path = args[0] #Path to the data folder

    import sys
    run(data_path=data_path)
    sys.exit()
