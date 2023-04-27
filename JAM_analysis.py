# Performs the analysis of results after the non-linear search.
# Several integrated quantities are measured, as well
# as their comparison with the reference values.



from optparse import OptionParser

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import json

from plotbin.plot_velfield import plot_velfield
from jampy.mge_radial_density import mge_radial_density
from jampy.mge_radial_mass import mge_radial_mass
from jampy.mge_half_light_isophote import mge_half_light_isophote
from astropy import units as u



def run(result_path, data_path, alpha=2.5):
    analysis_path = result_path+"Analysis/" # path where the analysis will be saved
    os.makedirs(analysis_path)              # create the analysis folder

        # Load sampler and model
    with open(result_path+f'/Final_sampler.pickle','rb') as f:
        sampler = pickle.load(f)
        f.close()
    with open(result_path+f'/JAM_class.pickle','rb') as f:
        Jam_Model = pickle.load(f)
        f.close()
    with open(result_path+f'/priors.pickle','rb') as f:
        priors = pickle.load(f)
        f.close()
    # Generate a new set of results with statistical+sampling uncertainties.
    labels = list(priors.keys())
    parsRes = priors.copy()
    results_sim = dyfunc.jitter_run(dyfunc.resample_run(sampler.results))
    samples_sim, weights = results_sim.samples, results_sim.importance_weights()
    quantiles = [dyfunc.quantile(samps, [0.16, 0.5,  0.84], weights=weights)
                for samps in samples_sim.T]                        #quantiles for each parameter
    
        #Update the parameters
    for i, key in enumerate(parsRes.keys()):
        parsRes[key] = quantiles[i][1]

        # Tracer plot
    fig, axes = dyplot.traceplot(results=results_sim, show_titles=True,
                             labels=labels,
                             )
    fig.tight_layout()
    plt.savefig(analysis_path+"/tracer_plot.png")
    plt.close()
    
        # Plot the 2-D marginalized posteriors.
    cfig, caxes, = dyplot.cornerplot(results_sim, smooth=0.08,
                                    show_titles=True,labels=labels,
                            )
    fig.tight_layout()
    plt.savefig(analysis_path+"/corner_plot.png")
    plt.close()

        # Data and model
    Jam_Model.Updt_Model(UpdtparsDic=parsRes)
    plt.figure(figsize=(18,10))
    rmsModel, ml, chi2, flux  = Jam_Model.run_current(quiet=False, 
                                                    plot=True,nodots=False, 
                                                    cmap="rainbow", label=r"km/s")
    plt.tight_layout()
    plt.savefig(analysis_path+"/model.png")
    plt.close()

        # Residual
    fig = plt.figure(figsize=(18, 10))
    plot_velfield(Jam_Model.xbin, Jam_Model.ybin, 
                100*abs(Jam_Model.rms-rmsModel)/Jam_Model.rms,
                colorbar=True,  cmap="rainbow",  
                markersize=0.2, label="[%]")
    plt.title(r"${\Delta V_{\rm rms}^{*}}$")

    fig.tight_layout()
    plt.savefig(analysis_path+"/residual.png")
    plt.close()

        # Creates a json file with the description of the measured quantities
    out_descripition = open("{}/description.json".format(analysis_path), "w")
    description = { "Reff":  "MGE effetive radius, in arcsec.",
                    "R":     "Radius where quantities were measured, in arcsec.",
                    "Mstar": "True stellar mass within R, in log10.", 
                    "Mdm":   "True dm mass within R, in log10.", 
                    "Mbh":   "True BH mass within R, in log10.", 
                    "Mtotal":"True total mass within R, in log10.", 
                    "fdm":   "Fraction of DM within R.",
                    "MMstar": "Model stellar mass within R, in log10.", 
                    "MMdm":   "Model dm mass within R, in log10.", 
                    "MMbh":   "Model BH mass within R, in log10.", 
                    "MMtotal":"Model total mass within R, in log10.", 
                    "Mfdm":   "Model DM fraction within R.",
                    "Dstar":  "(MMstar - Mstar)/Mstar",
                    "Ddm":    "(MMdm - Mdm)/Mdm",
                    "Dtotal": "(MMtotal - Mtotal)/Mtotal",
                    "Dfdm":   "(Mfdm - fdm)/fdm"
                }
    json.dump(description, out_descripition, indent = 8)
    out_descripition.close()

    # Get the effective radius in arcsec, and other quantities
    # See mge_half_light_isophote documentation for more details
    reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(Jam_Model.surf_lum,
                                                Jam_Model.sigma_lum,
                                                Jam_Model.qobs_lum,
                                                Jam_Model.distance)
    
    #  Model quantities
    R     = alpha*reff    # alpha times the Reff in arcsec
    R_kpc = ( (R*u.arcsec * Jam_Model.distance*u.Mpc ).to(
                        u.kpc,u.dimensionless_angles()) ).value # value in kpc


        # Get the radial mass of stars and DM within R
    MMstar =  mge_radial_mass(Jam_Model.surf_lum * Jam_Model.ml_model, 
                                Jam_Model.sigma_lum, Jam_Model.qobs_lum,
                                Jam_Model.inc, R, Jam_Model.distance)

    MMdm = mge_radial_mass(Jam_Model.surf_dm, 
                                Jam_Model.sigma_dm, Jam_Model.qobs_dm,
                                Jam_Model.inc, R, Jam_Model.distance)
    MMbh = Jam_Model.mbh # Model BH mass
    MMtotal = MMstar + MMdm + MMbh # Total Mass 
    Mfdm = MMdm / MMtotal          # Dark matter fraction

    # Data quantities

    # Load the snapshot data
    dm_dataset   = np.load(data_path+"/dm/coordinates_dark.npy")
    star_dataset = np.load(data_path+"/imgs/coordinates_star.npy")
        
        # Stellar content
    rStar = np.sqrt(np.sum(star_dataset[:, 0:3]**2, axis=1)) # radius
    i = rStar <= R_kpc                                       # only particles within R
    Mstar = sum(star_dataset[:, 6][i]*1e10)                  # mass within   R

        # Dark content
    rDM = np.sqrt(np.sum(dm_dataset[:, 0:3]**2, axis=1))
    i = rDM <= R_kpc
    Mdm = sum(dm_dataset[:,3][i]*1e10)

    Mbh = (0.015411)*(1e10/0.67) # BH mass from TNG catalogue
    Mtotal = Mstar + Mdm + Mbh   # Total snapshot mass within R
    fdm = Mdm/Mtotal             # DM fraction in the snapshot within R
    
    print('=' * term_size.columns)
        # Accuracy in stellar mass
    print("Model stellar Mass: {:.2e} Msun".format( float(MMstar) ))
    print("Data  stellar Mass: {:.2e} Msun".format( float(Mstar) ))
    Dstar = float ( (MMstar - Mstar)/Mstar )
    print("(Model - Data)/Data: {:.2f}".format( float(Dstar) ))
    print('=' * term_size.columns)
        # Accuracy in Dm mass
    print("Model dm Mass: {:.2e} Msun".format( float(MMdm) ))
    print("Data  dm Mass: {:.2e} Msun".format( float(Mdm) ))
    Ddm = float ( (MMdm - Mdm)/Mdm )
    print("(Model - Data)/Data: {:.2f}".format( float(Ddm) ))
    print('=' * term_size.columns)
        # Accuracy in total mass
    print("Model total Mass: {:.2e} Msun".format( float(MMtotal) ))
    print("Data  total Mass: {:.2e} Msun".format( float(Mtotal) ))
    Dtotal = float ( (MMtotal - Mtotal)/Mtotal )
    print("(Model - Data)/Data: {:.2f}".format( float(Dtotal) ))
    print('=' * term_size.columns)
        # Accuracy in dm fraction
    print("Model DM fraction: {:.2f}".format( float(Mfdm) ))
    print("Data  DM fraction: {:.2f}".format( float(fdm) ))
    Dfdm = float ( (Mfdm - fdm)/fdm )
    print("(Model - Data)/Data: {:.2f}".format( float(Dfdm) ))

        # Radial density profiles
    radii    = np.arange(0.1, 10*reff, 0.01)   # Radii in arcsec
    pc       = Jam_Model.distance*np.pi/0.648  # Constant factor to convert arcsec --> pc
    radii_pc = radii*pc                        # Radii in pc

    dstar = mge_radial_density(Jam_Model.surf_lum * Jam_Model.ml_model, 
                                Jam_Model.sigma_lum, Jam_Model.qobs_lum,
                                Jam_Model.inc, radii, Jam_Model.distance)

    ddm = mge_radial_density(Jam_Model.surf_dm, 
                                Jam_Model.sigma_dm, Jam_Model.qobs_dm,
                                Jam_Model.inc, radii, Jam_Model.distance)

    # Load DM density profile
    from astropy.io import fits
    dm_hdu = fits.open(data_path+"/dm/density_fit.fits")
    true_density = dm_hdu[1].data["density"]
    true_radii = dm_hdu[1].data["radius"]
    dm_fit = dm_hdu[1].data["bestfit"]

    i = true_radii < radii_pc.max()

    true_density = true_density[i]
    true_radii   = true_radii[i]
    dm_fit = dm_fit[i]

    # Load star density profile
    star_hdu = fits.open(data_path+"/imgs/stellar_density.fits")
    rho_stars = star_hdu[1].data["density"]
    r_star = star_hdu[1].data["radius"]


    i = r_star < radii_pc.max()

    rho_stars = rho_stars[i]
    r_star   = r_star[i]

    plt.rcParams['xtick.labelsize']= 12
    plt.rcParams['ytick.labelsize']= 12


    plt.figure(figsize=(15,8))

    plt.plot(radii_pc, np.log10(dstar), label="Star", color="red")
    plt.plot(radii_pc, np.log10(ddm), label="DM", color="magenta")
    plt.plot(radii_pc, np.log10(ddm+dstar), label="Total", color="black")
    plt.plot(true_radii, np.log10(dm_fit), label="DM Fit", color="blue", markersize=12)
    #plt.plot(radii_pc, np.log10(a), label="MGE", color="black")

    plt.plot(true_radii, np.log10(true_density),  ".", label="DM data", color="magenta", markersize=12)

    plt.plot(r_star, np.log10(rho_stars),  ".", label="Stars data", color="red", markersize=12)
    plt.plot(r_star, np.log10(rho_stars+true_density),  ".", label="Total data", color="black", markersize=12)




    plt.xlabel("radii [pc]", size=20)
    plt.ylabel("$\log_{10}(\\frac{\\rho}{M_\odot/pc^3})$", size=20)
    plt.legend()
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(analysis_path+"/density_profiles.png")
    plt.close()


    # Save quantities
    r = {}
    r["Reff"]   = reff
    r["R"]      = R
    r["Mstar"]  = float( np.log10(Mstar) )
    r["Mdm"]    = float( np.log10(Mdm)   )
    r["Mbh"]    = float( np.log10(Mbh)   )
    r["Mtotal"] = float( np.log10(Mtotal) )
    r["fdm"]    = fdm
    r["MMstar"]  = float( np.log10(MMstar) )
    r["MMdm"]    = float( np.log10(MMdm)   )
    r["MMbh"]    = float( np.log10(MMbh)   )
    r["MMtotal"] = float( np.log10(MMtotal) )
    r["Mfdm"]    = float( Mfdm )
    r["Dstar"]   = Dstar
    r["Ddm"]     = Ddm
    r["Dtotal"]  = Dtotal
    r["Dfdm"]    = Dfdm
    # the json file where the output must be stored
    out_r = open("{}/quantities.json".format(analysis_path), "w")
    json.dump(r, out_r, indent = 8)
    out_r.close()



if __name__ == '__main__':
    term_size = os.get_terminal_size()
    parser = OptionParser() #You also should inform the folder name
    parser.add_option('--alpha', action='store', type=float, dest='alpha',
                      default=2.5, 
                      help='Fraction of the effective radius where the quantities will be measured.')

    (options, args) = parser.parse_args()
    if len(args) != 2:
        print('Error - please provide the paths to results and data.')
        sys.exit(1)
    result_path = args[0] # Path to the results folder
    data_path   = args[1] # Path to the data folder

    import sys
    run(result_path=result_path, data_path=data_path, alpha=options.alpha)
    sys.exit()
