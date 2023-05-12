# Performs the analysis of results after the non-linear search.
# Several integrated quantities are measured, as well
# as their comparison with the reference values.



from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12

import json
import pickle
import os


from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import autolens as al
import autolens.plot as aplt

from plotbin.plot_velfield import plot_velfield
from jampy.mge_radial_mass import mge_radial_mass
from jampy.mge_radial_density import mge_radial_density
from jampy.mge_half_light_isophote import mge_half_light_isophote
from astropy import units as u



def run(result_path, data_path, phase_name, alpha=None, Re=None):
    result_path   = result_path+"/"+phase_name  #path to the non-linear results
    if Re:
        analysis_path = result_path+"/Analysis_Re/"                       # path where the analysis will be saved
    else:
        analysis_path = result_path+"/Analysis_{:.1f}Reff/".format(alpha) # path where the analysis will be saved
    os.makedirs(analysis_path)               # create the analysis folder

    with open(result_path+'/Final_sampler_{}.pickle'.format(phase_name),'rb') as f:
        sampler = pickle.load(f)
        f.close()

    with open(result_path+'/{}.pickle'.format(phase_name),'rb') as f:
        phase = pickle.load(f)
        f.close()

    with open(result_path+'/CombinedModel_{}.pickle'.format(phase_name),'rb') as f:
        CM = pickle.load(f)
        f.close()
        
    with open(result_path+'/priors_{}.pickle'.format(phase_name),'rb') as f:
        priors = pickle.load(f)
        f.close()
    
    sampler = sampler["sampler"]
    sampler.results.summary()

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

    if phase_name == "phase1":
        source_ell_comp = al.convert.elliptical_comps_from(axis_ratio=parsRes["source_q"], 
                                                        angle=parsRes["source_phi"])

        source_model = al.Galaxy(
            redshift=CM.Lens_model.z_s,
            light=al.lp.EllSersic(
                centre=(parsRes["source_y0"], parsRes["source_x0"]),
                elliptical_comps=source_ell_comp,
                intensity=parsRes["source_intensity"],
                effective_radius=parsRes["source_eff_r"],
                sersic_index=parsRes["source_n_index"],
            ),
        )    


        CM.source_galaxy(source_model)

    else:
        
        adp_pix = phase.source_pix(pixels=int(phase.parsSource["pixels"]),
                                    weight_floor=phase.parsSource["weight_floor"],
                                    weight_power=phase.parsSource["weight_power"]
                                    )

        adp_reg = phase.source_reg(inner_coefficient=phase.parsSource["inner_coefficient"],
                                    outer_coefficient=phase.parsSource["outer_coefficient"],
                                    signal_scale=phase.parsSource["signal_scale"]
                                    )

        source_model = al.Galaxy(redshift=CM.Lens_model.z_s,
                                    pixelization=adp_pix, regularization=adp_reg,
                                    hyper_model_image=phase.hyper_image_2d,
                                    hyper_galaxy_image=phase.hyper_image_2d,
                                    )

        CM.source_galaxy(source_model=source_model)  #Setting the source galaxy model
    
    CM.quiet = True
    CM.Updt_Model(parsRes)
    print("Generating lensing results. This could take a while.")

    # Config. Pyautolens plots
    cmap = aplt.Cmap(cmap="rainbow")

    mat_plot_2d_output = aplt.MatPlot2D(
        output=aplt.Output(
            filename=f"None",
            path=analysis_path,
            format=["png",],
            format_folder=True,
        ),
        cmap=cmap)
    mat_plot_2d =  mat_plot_2d_output
    # Make a fit plotter
    fit_plotter = aplt.FitImagingPlotter(fit=CM.Fit, mat_plot_2d=mat_plot_2d)
    
    mat_plot_2d.output.filename = "fit_subplot"
    fit_plotter.subplot_fit_imaging()
    
    mat_plot_2d.output.filename = "fit_image"
    fit_plotter.figures_2d(image=True)

    mat_plot_2d.output.filename = "fit_model"
    fit_plotter.figures_2d(model_image=True)

    mat_plot_2d.output.filename = "residual"
    fit_plotter.figures_2d(residual_map=True)

    mat_plot_2d.output.filename = "subplot_plane"
    fit_plotter.subplot_of_planes(plane_index=1)
    
    # Make a inversion plotter
    inversion_plotter = aplt.InversionPlotter(inversion=CM.Fit.inversion, 
                                            mat_plot_2d=mat_plot_2d)

    mat_plot_2d.output.filename = "reconstruction"
    inversion_plotter.figures_2d_of_mapper(mapper_index=0,
                        reconstruction=True)
    
    mat_plot_2d.output.filename = "subplot_inversion"
    inversion_plotter.subplot_of_mapper(mapper_index=0)

    # Plot dynamical model and residual
    fig = plt.figure(figsize=(18, 10))
    rmsModel, ml, chi2, chi2T = CM.Jampy_model._run(plot=True, cmap="rainbow", label=r"km/s", xlabel="arcsec")

    plt.tight_layout()
    plt.savefig(analysis_path+"/jam_model.png")
    plt.close()

    fig = plt.figure(figsize=(18, 10))
    plot_velfield(CM.Jampy_model.xbin, CM.Jampy_model.ybin, 
                100*abs(CM.Jampy_model.rms-rmsModel)/CM.Jampy_model.rms,
                colorbar=True,  cmap="rainbow",  
                markersize=0.2, label="[%]")
    plt.title(r"${\Delta V_{\rm rms}^{*}}$")

    fig.tight_layout()
    plt.savefig(analysis_path+"/jam_residual.png")
    plt.close()
    
    # Creates a json file with the description of the measured quantities
    out_descripition = open("{}/description.json".format(analysis_path), "w")
    description = { "Reff":  "MGE effetive radius, in arcsec.",
                    "thetaE": "Measured Einstein Ring in arcsec",
                    "R":     "Radius where quantities were measured, in arcsec.",
                    "Mstar": "True stellar mass within R, in 1e10Msun.", 
                    "Mdm":   "True dm mass within R, in 1e10Msun.", 
                    "Mbh":   "True BH mass within R, in 1e10Msun.", 
                    "Mtotal":"True total mass within R, in 1e10Msun.", 
                    "fdm":   "Fraction of DM within R.",
                    "MMstar": "Model stellar mass within R, in 1e10Msun.", 
                    "MMdm":   "Model dm mass within R, in 1e10Msun.", 
                    "MMbh":   "Model BH mass within R, in 1e10Msun.", 
                    "MMtotal":"Model total mass within R, in 1e10Msun.", 
                    "Mfdm":   "Model DM fraction within R.",
                    "Dstar":  "(MMstar - Mstar)/Mstar",
                    "Ddm":    "(MMdm - Mdm)/Mdm",
                    "Dtotal": "(MMtotal - Mtotal)/Mtotal",
                    "Dfdm":   "(Mfdm - fdm)/fdm"
                }
    json.dump(description, out_descripition, indent = 8)
    out_descripition.close()

    einstein_radius = CM.Fit.tracer.einstein_radius_from(CM.Fit.imaging.grid)
    # Get the effective radius in arcsec, and other quantities
    # See mge_half_light_isophote documentation for more details
    reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(CM.Lens_model.surf_lum,
                                                            CM.Lens_model.sigma_lum,
                                                            CM.Lens_model.qobs_lum,
                                                            CM.Jampy_model.distance)
    #  Model quantities
    if Re:
        R = einstein_radius # Einstein ring in arcsec
    else: 
        R = alpha*reff      # alpha times the Reff in arcsec

    R_kpc = ( (R*u.arcsec * CM.Jampy_model.distance*u.Mpc ).to(
                        u.kpc,u.dimensionless_angles()) ).value # 2.5Reff in kpc


        # Get the radial mass of stars and DM within rad
    MMstar = float ( mge_radial_mass(CM.Jampy_model.surf_lum * CM.Jampy_model.ml_model, 
                                CM.Jampy_model.sigma_lum, CM.Jampy_model.qobs_lum,
                                CM.Jampy_model.inc, R, CM.Jampy_model.distance) )

    MMdm = float ( mge_radial_mass(CM.Jampy_model.surf_dm, 
                                CM.Jampy_model.sigma_dm, CM.Jampy_model.qobs_dm,
                                CM.Jampy_model.inc, R, CM.Jampy_model.distance) )
    MMbh = float ( CM.Jampy_model.mbh )  # Model BH mass

        # Total Mass
    MMtotal = MMstar + MMdm
        # Dark matter fraction
    Mfdm = MMdm / MMtotal

    # Data quantities
    info = fits.open(data_path+"/imgs/log_img.fits")[1].data

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
    if info["logMbh"].size != 1: #If there is more than one BH
        Mbh = list(np.float_(info["logMbh"][0])) # Sould be the BH mass from TNG catalogue 
    else:
        Mbh = float( info["logMbh"] ) # Sould be the BH mass from TNG catalogue 
    Mtotal = Mstar + Mdm        # Total snapshot mass within R
    fdm    = Mdm/Mtotal         # DM fraction in the snapshot within R

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
    pc       = CM.Jampy_model.distance*np.pi/0.648  # Constant factor to convert arcsec --> pc
    radii_pc = radii*pc                        # Radii in pc

    dstar = mge_radial_density(CM.Jampy_model.surf_lum * CM.Jampy_model.ml_model, 
                                CM.Jampy_model.sigma_lum, CM.Jampy_model.qobs_lum,
                                CM.Jampy_model.inc, radii, CM.Jampy_model.distance)

    ddm = mge_radial_density(CM.Jampy_model.surf_dm, 
                                CM.Jampy_model.sigma_dm, CM.Jampy_model.qobs_dm,
                                CM.Jampy_model.inc, radii, CM.Jampy_model.distance)

    # Load DM density profile
    dm_hdu       = fits.open(data_path+"/dm/density_fit.fits")
    true_density = dm_hdu[1].data["density"]
    true_radii   = dm_hdu[1].data["radius"]
    dm_fit       = dm_hdu[1].data["bestfit"]

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
    r["R"]      = float( R )
    r["thetaE"] = float(einstein_radius )
    r["Mstar"]  = float( np.log10(Mstar) )
    r["Mdm"]    = float( np.log10(Mdm)   )
    r["Mbh"]    = Mbh
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
                      default=None, 
                      help='Fraction of the effective radius where the quantities will be measured.')
    parser.add_option('--Re', action='store_true',  dest='Re',
                      default=False, 
                      help='Einstein radius, in arcsec, where the quantities are measured.')
    parser.add_option('--phase', action='store', type=str, dest='phase',
                      default="phase5", 
                      help='Phase to be analysed.')

    (options, args) = parser.parse_args()
    if len(args) != 2:
        print('Error - please provide the paths to results and data.')
        sys.exit(1)
    
    if options.alpha and options.Re:
        print('alpha and Re parameters cannot be settled at the same time.')
        sys.exit(1)
    elif (not options.alpha) and (not options.Re):
        print('You must set alpha or Re.')
        sys.exit(1)
    else: pass

    result_path = args[0] # Path to the results folder
    data_path   = args[1] # Path to the data folder

    import sys
    run(result_path=result_path, data_path=data_path, 
                    phase_name=options.phase, alpha=options.alpha, Re=options.Re)
    sys.exit()
