# ## Testing the a full Pipeline
# 
# Model: DM parametrized by gNFW, black hole, scalar anisotropy, 
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


def run(data_path, ncores):
        #Reading inputs
    log = fits.open('{}/log_img.fits'.format(data_path))[1].data

        #MGE components
    sol_star, pa, eps, xmed, ymed = np.load('%s/mge.npy'%data_path,  allow_pickle=True)

        #Lens/cosmology info
    info  = fits.open('{}/image.fits'.format(data_path))[0].header
    cosmo = FlatLambdaCDM(H0=info["H0"], Om0=info["OMEGA0"])
    pixel_scale = info["PIXSCALE"]
    
    noise = fits.open('{}/noise.fits'.format(data_path))[0].data
    r = dist_circle(noise.shape[1]/2, 
                    noise.shape[0]/2, 
                    noise.shape)          # distance matrix from the centre
    mask = r <= 0.25/pixel_scale          # pixels within 0.25'' radii
    noise[mask] = noise[mask]*10          # scale the noise within mask by a factor of 10
    fits.writeto('{}/noise_scaled.fits'.format(data_path), data=noise, 
                 overwrite=True) # save scaled noise

        #Lens data
    
    imaging = al.Imaging.from_fits(
        image_path=path.join("./{}/".format(data_path), "image.fits"),
        noise_map_path=path.join("./{}/".format(data_path), "noise_scaled.fits"),
        psf_path=path.join("./{}/".format(data_path), "psf.fits"),
        pixel_scales=pixel_scale,
    )
    
    
    
    mask_2d = al.Mask2D.circular(shape_native=imaging.shape_native,
                                        radius=2.5,
                                        pixel_scales=pixel_scale)
    imaging = imaging.apply_mask(mask_2d)

        #Combined model building
    z_l     = info["z_l"]                                                  #lens Redshift
    z_s     = info["z_s"]                                                  #source Redshift

    surf  = sol_star[:,0]
    sigma = sol_star[:,1]
    qObs  = sol_star[:,2]

    # Lens model
    mge_mass_profile = Lens.mass_profile.MGE()  #initialize the class
        
        #Setting the parameters
    mge_mass_profile.MGE_comps(z_l=z_l, z_s=z_s, 
                                surf_lum=surf, sigma_lum=sigma, qobs_lum=qObs,
                                )
        #Add DM component
    ellgNFW  = al.mp.EllNFWGeneralized()

    mge_mass_profile.Analytic_Model(ellgNFW) 

    CM = CombinedModel.Model(Jampy_model=None, 
                            Lens_model=mge_mass_profile, 
                            masked_imaging=imaging, cosmology=cosmo, quiet=True)

    #Setup Configurations
    CM.set_mass_to_light(ml_kind='scalar')          #Setting scalar ML
    CM.include_DM_MGE(profile="gNFW")               #Setting Dark matter component
    CM.include_DM_analytical(analytical_DM=ellgNFW)  #Analytical eNFW

    #Build the model, and passing the priors
        # minumum inclination allowed
    parsDic = {
                "ml":      p.UniformPrior(1.0, 10),
                "log_rho_s":  p.UniformPrior(-6, 0),
                "rs":      p.UniformPrior(1e-2, 30),
                "qDM":     1.0,
                "slope":   p.UniformPrior(0.5, 2.0),
                "gamma":   1.0,
                }


    CM.build_model(parsDic)

        ## Initialize the phase1
    from dyLens.pipelines.phase_1 import Ph1Model
    output_path = data_path.split("/")[1]+"/model3"
    ph1 = Ph1Model(CombinedModel=CM, output_path=output_path)

        ## Update the non-linear sampler to save some time
    ph1.config_non_linear(n_cores=ncores, nlive=180)

        #Run Phase1 nested sampler
    ph1.run_dynesty(maxiter=300)

    from dyLens.pipelines.phase_2 import Ph2Model

    ph2 = Ph2Model(output_path=output_path, instance=ph1)
    ph2.config_non_linear(n_cores=ncores)
    ph2.run_dynesty(maxiter=300)

    from dyLens.pipelines.phase_3 import Ph3Model

    ph3 = Ph3Model(output_path=output_path, instance=ph2)
    ph3.config_non_linear(n_cores=ncores, nlive=500)
    ph3.run_dynesty(maxiter=300)

    from dyLens.pipelines.phase_4 import Ph4Model

    ph4 = Ph4Model(output_path=output_path, instance=ph3)
    ph4.config_non_linear(n_cores=ncores)
        ## Update the source galaxy priors based on the simulated data
    New_sourcePriors = {"pixels": p.UniformPrior(10, 300) }
    ph4.config_pixelized_source(New_sourcePriors)

    ph4.run_dynesty(maxiter=300)

    from dyLens.pipelines.phase_5 import Ph5Model

    ph5 = Ph5Model(output_path=output_path, instance=ph4, 
                    n_likelihoods=350, n_cores=ncores, threshold=0.2)
    ph5.config_non_linear(n_cores=ncores, nlive=500)
    ph5.run_dynesty(maxiter=300)

if __name__ == '__main__':
    parser = OptionParser() #You also should inform the folder name
    parser.add_option('--ncores', action='store', type='int', dest='ncores',
                      default=8, help='number of cores')

    (options, args) = parser.parse_args()
    if len(args) != 1:
        print('Error - please provide the path to the date.')
        sys.exit(1)
    data_path = args[0] #Path to the data folder

    import sys
    print("Number of cores:", options.ncores)
    print("\n")
    run(data_path=data_path, ncores=options.ncores)
    sys.exit()
