# Performs the analysis of results after the non-linear search.
# Several integrated quantities are measured, as well
# as their comparison with the reference values.


import sys
from optparse import OptionParser

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12

import pickle
import os
import json

from plotbin.plot_velfield import plot_velfield
from jampy.mge_radial_density import mge_radial_density
from jampy.mge_radial_mass import mge_radial_mass
from jampy.mge_half_light_isophote import mge_half_light_isophote
from astropy import units as u

import autolens as al
import autolens.plot as aplt

from util import quantities2D, quantities3D
from dyLens.utils.tools import effective_einstein_radius_from_kappa, mge_radial_mass2d
from dyLens.Combined import updt_model
from copy import deepcopy



def _JAM_figs(Jam_Model, parsRes, analysis_path):
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

def _dyLens_Lens_figs(phase, CM, parsRes, phase_name, analysis_path, dyLens=None):
    
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
    
    if dyLens: # Only if the analysis is a dyLens analysis
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
    else: pass

def _JAM_analysis(Jam_Model, R):
            # 3D quantities
        # Get the radial mass of stars and DM within R
    MMstar =  mge_radial_mass(Jam_Model.surf_lum * Jam_Model.ml_model, 
                                Jam_Model.sigma_lum, Jam_Model.qobs_lum,
                                Jam_Model.inc, R, Jam_Model.distance)

    MMdm = mge_radial_mass(Jam_Model.surf_dm, 
                                Jam_Model.sigma_dm, Jam_Model.qobs_dm,
                                Jam_Model.inc, R, Jam_Model.distance)
    MMbh = Jam_Model.mbh     # Model BH mass
    MMtotal = MMstar + MMdm  # Total Mass 
    Mfdm = MMdm / MMtotal    # Dark matter fraction

            # 2D quantities
        # Get the radial projected mass of stars and DM within R
    projMMstar =  mge_radial_mass2d(Jam_Model.surf_lum * Jam_Model.ml_model,
                                Jam_Model.sigma_lum, Jam_Model.qobs_lum,
                                a=0, b=R, distance=Jam_Model.distance)

    projMMdm = mge_radial_mass2d(Jam_Model.surf_dm,
                                Jam_Model.sigma_dm, Jam_Model.qobs_dm,
                                a=0, b=R, distance=Jam_Model.distance)
    projMMtotal = projMMstar + projMMdm  # Total projected Mass 
    projMfdm = projMMdm / projMMtotal    # Projected Dark matter fraction

    return MMstar, MMdm, MMbh, MMtotal, Mfdm, projMMstar, projMMdm, projMMtotal, projMfdm

def _dyLens_Lens_analysis(CM, R, dyLens=None):
    if dyLens:
            # 3D quantities
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

            # 2D quantities
        # Get the radial projected mass of stars and DM within R
        projMMstar =  mge_radial_mass2d(CM.Lens_model.surf_lum * CM.ml_model,
                                    CM.Lens_model.sigma_lum, CM.Lens_model.qobs_lum,
                                    a=0, b=R, 
                                    z=CM.Lens_model.z_l, cosmology=CM.cosmology)

        projMMdm = mge_radial_mass2d(CM.Jampy_model.surf_dm,
                                    CM.Jampy_model.sigma_dm, CM.Jampy_model.qobs_dm,
                                    a=0, b=R, 
                                    z=CM.Lens_model.z_l, cosmology=CM.cosmology)
        projMMtotal = projMMstar + projMMdm  # Total projected Mass 
        projMfdm = projMMdm / projMMtotal    # Projected Dark matter fraction


    else:
            # 3D quantities
        MMstar = MMdm = MMbh = MMtotal = Mfdm = 0.0  # Lens only is not sensitive to them.
            # 2D quantities
        # Get the radial projected mass of stars and DM within R
        projMMstar =  mge_radial_mass2d(CM.Lens_model.surf_lum * CM.ml_model,
                                    CM.Lens_model.sigma_lum, CM.Lens_model.qobs_lum,
                                    a=0, b=R, 
                                    z=CM.Lens_model.z_l, cosmology=CM.cosmology)
            
            # Just to get the DM parametrized by gaussians
        CM_dm = deepcopy(CM)
        CM_dm.include_DM_MGE(profile=CM_dm.dm_profile_name)
        updt_model.Updt_DM_MGE(CM_dm)

        projMMdm = mge_radial_mass2d(CM_dm.surf_dm_model,
                                    CM_dm.sigma_dm_model,
                                    np.full_like(CM_dm.surf_dm_model, CM.parsDic["qDM"]),
                                    a=0, b=R,
                                    z=CM.Lens_model.z_l, cosmology=CM.cosmology)
        projMMtotal = projMMstar + projMMdm  # Total projected Mass 
        projMfdm = projMMdm / projMMtotal    # Projected Dark matter fraction
        
        
        # kappa MGE model. Uses the same grid as the lens image
    kappa_model = CM.Lens_model.convergence_2d_from(CM.Fit.imaging.unmasked.grid)

        # Calculates model the Einstein ring
        # Uses the same grid as the lens image
    Re_model = effective_einstein_radius_from_kappa(kappa_model, 
                                                    grid_spacing=CM.Fit.image.pixel_scale,
                                                    grid=CM.Fit.imaging.unmasked.grid, 
                                                    Nsamples=CM.Fit.imaging.unmasked.grid.shape[0]/2)

    return MMstar, MMdm, MMbh, MMtotal, Mfdm, projMMstar, projMMdm, projMMtotal, projMfdm, Re_model

def _3D_density_profile(Jam_Model, reff, data_path, analysis_path, ):
    # Radial density profiles
    radii    = np.arange(0.1, 10*reff, 0.01)   # Radii in arcsec
    pc       = Jam_Model.distance*np.pi/0.648  # Constant factor to convert arcsec --> pc
    radii_pc = radii*pc                        # Radii in pc
    reff_pc = reff*pc

    dstar = mge_radial_density(Jam_Model.surf_lum * Jam_Model.ml_model, 
                                Jam_Model.sigma_lum, Jam_Model.qobs_lum,
                                Jam_Model.inc, radii, Jam_Model.distance)

    ddm = mge_radial_density(Jam_Model.surf_dm, 
                                Jam_Model.sigma_dm, Jam_Model.qobs_dm,
                                Jam_Model.inc, radii, Jam_Model.distance)

    # Load DM density profile
    dm_hdu = fits.open(data_path+"/dm/density_fit.fits")
    true_density = dm_hdu[1].data["density"]
    true_radii   = dm_hdu[1].data["radius"]
    dm_fit       = dm_hdu[1].data["bestfit"]

    i = true_radii < radii_pc.max()

    true_density = true_density[i]
    true_radii   = true_radii[i]
    dm_fit = dm_fit[i]

    # Load star density profile
    star_hdu  = fits.open(data_path+"/imgs/stellar_density.fits")
    rho_stars = star_hdu[1].data["density"]
    r_star    = star_hdu[1].data["radius"]


    i = r_star < radii_pc.max()

    rho_stars = rho_stars[i]
    r_star    = r_star[i]

    plt.figure(figsize=(15,8))

    plt.plot(radii_pc, np.log10(dstar), label="Star", color="red")
    plt.plot(radii_pc, np.log10(ddm), label="DM", color="magenta")
    plt.plot(radii_pc, np.log10(ddm+dstar), label="Total", color="black")
    plt.plot(true_radii, np.log10(dm_fit), label="DM Fit", color="blue", markersize=12)
    plt.axvline(reff_pc, label="$R_{eff}$", linestyle="--", markersize=12)

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

def run(result_path, data_path, JAM=None, dyLens=None, Lens=None,
            alpha=None, Re=None, phase_name=None):
    
    if JAM == dyLens == Lens == None:
        raise ValueError("You must provide at least one of the keywords related to the analysis: JAM, Lens or Lens")
    else:
        if (JAM and Lens) or (JAM and dyLens):
            raise ValueError("dyLens/Lens analysis is not possible with JAM analysis")
        elif dyLens or Lens:
            assert phase_name != None, "You must provide phase name to be analysed"
            result_path   = result_path+"/"+phase_name  # path to the non-linear results
        else: pass

    if Re:
        analysis_path = result_path+"/Analysis_Re/"                       # path where the analysis will be saved
    else:
        analysis_path = result_path+"/Analysis_{:.1f}Reff/".format(alpha) # path where the analysis will be saved
    os.makedirs(analysis_path)               # create the analysis folder
    
    if JAM: 
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
    
    else:
            # Load sampler, phase and model of dyLens/Lens
        with open(result_path+'/Final_sampler_{}.pickle'.format(phase_name),'rb') as f:
            sampler = pickle.load(f)
            f.close()
        sampler = sampler["sampler"]

        with open(result_path+'/{}.pickle'.format(phase_name),'rb') as f:
            phase = pickle.load(f)
            f.close()

        with open(result_path+'/CombinedModel_{}.pickle'.format(phase_name),'rb') as f:
            CM = pickle.load(f)
            f.close()
            
        with open(result_path+'/priors_{}.pickle'.format(phase_name),'rb') as f:
            priors = pickle.load(f)
            f.close()

    # Generate a new set of results with statistical+sampling uncertainties.
    sampler.results.summary()
    labels = list(priors.keys())
    parsRes = priors.copy()
    results_sim = dyfunc.jitter_run(dyfunc.resample_run(sampler.results))
    samples_sim, weights = results_sim.samples, results_sim.importance_weights()
    quantiles = [dyfunc.quantile(samps, [0.16, 0.5,  0.84], weights=weights)
                for samps in samples_sim.T]                        #quantiles for each parameter
    
        # Update the parameters
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
    
    if JAM:
        _JAM_figs(Jam_Model=Jam_Model, parsRes=parsRes, 
                            analysis_path=analysis_path)
    else:
        _dyLens_Lens_figs(phase=phase, CM=CM, parsRes=parsRes, 
                            phase_name=phase_name, 
                            analysis_path=analysis_path, 
                            dyLens=dyLens)
       
        # Creates a json file with the description of the measured quantities
    out_descripition = open("{}/description.json".format(analysis_path), "w")
    description = { "Reff":  "MGE effetive radius, in arcsec.",
                    "R":     "Radius where quantities were measured, in arcsec.",
                    "MthetaE": "True Einstein Ring in arcsec",
                    "Mstar": "True stellar mass within R, in log10.", 
                    "Mdm":   "True dm mass within R, in log10.", 
                    "Mbh":   "True BH mass within R, in log10.", 
                    "Mtotal":"True total mass within R, in log10.", 
                    "fdm":   "Fraction of DM within R.",
                    "projMstar": "Projected true stellar mass within R, in log10.", 
                    "projMdm":   "Projected true dm mass within R, in log10.", 
                    "projMtotal":"Projected true total mass within R, in log10.", 
                    "projfdm":   "Projected fraction of DM within R.",
                    "MMstar": "Model stellar mass within R, in log10.", 
                    "MMdm":   "Model dm mass within R, in log10.", 
                    "MMbh":   "Model BH mass within R, in log10.", 
                    "MMtotal":"Model total mass within R, in log10.", 
                    "Mfdm":   "Model DM fraction within R.",
                    "MMthetaE": "Measured Einstein Ring in arcsec",
                    "projMMstar": "Projected model stellar mass within R, in log10.", 
                    "projMMdm":   "Projected model dm mass within R, in log10.", 
                    "projMMtotal":"Projected model total mass within R, in log10.", 
                    "projMfdm":   "Projected model DM fraction within R.",
                    "Dstar":  "(MMstar - Mstar)/Mstar",
                    "Ddm":    "(MMdm - Mdm)/Mdm",
                    "Dtotal": "(MMtotal - Mtotal)/Mtotal",
                    "Dfdm":   "(Mfdm - fdm)/fdm",
                    "projDstar":  "proj (MMstar - Mstar)/Mstar",
                    "projDdm":    "proj (MMdm - Mdm)/Mdm",
                    "projDtotal": "proj (MMtotal - Mtotal)/Mtotal",
                    "projDfdm":   "proj (Mfdm - fdm)/fdm",
                    "DthetaE": "(MthetaE - MMthetaE)/MthetaE"
                }
    json.dump(description, out_descripition, indent = 8)
    out_descripition.close()
    
    if JAM:
        # Get the effective radius in arcsec, and other quantities
        # See mge_half_light_isophote documentation for more details
        
        distance = Jam_Model.distance # Distance in Mpc
        reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(Jam_Model.surf_lum,
                                                    Jam_Model.sigma_lum,
                                                    Jam_Model.qobs_lum,
                                                    distance)
    else: 
        
        distance = CM.cosmology.angular_diameter_distance(z=CM.Lens_model.z_l).value # Distance in Mpc
        reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(CM.Lens_model.surf_lum,
                                                    CM.Lens_model.sigma_lum,
                                                    CM.Lens_model.qobs_lum,
                                                    distance)
    
    #  Model quantities
    if dyLens or Lens:
        # Einstein ring 
            # surface mass density in 1e10 Msun/kpc2
        sigma_M    = fits.open(data_path+"/imgs/surface_mass_density.fits")[0]        
            # kappa from data. The critical density is in Msun/Mpc2, so I convert it to Msun/kpc2
        kappa_data = (sigma_M.data*1e10)/ ( CM.Lens_model.critical_density/(1e3)**2 )
        Re_data  = effective_einstein_radius_from_kappa(kappa_data, 
                                                        grid_spacing=CM.Fit.image.pixel_scale,
                                                        grid=CM.Fit.imaging.unmasked.grid)
    else: pass

    if Re:
        if JAM:
            raise ValueError("Jampy model is not able to infer the Einstein radius alone.")
        R = Re_data          # Einstein ring in arcsec
    else: 
        R = alpha*reff  # alpha times the Reff in arcsec
    R_kpc = ( (R*u.arcsec * distance*u.Mpc ).to(
                        u.kpc,u.dimensionless_angles()) ).value # value in kpc

    if JAM:
        MMstar, MMdm, MMbh, MMtotal, Mfdm, \
            projMMstar, projMMdm, projMMtotal, projMfdm = _JAM_analysis(Jam_Model=Jam_Model, R=R)
        Re_model = Re_data = DthetaE = 0
    else:
        MMstar, MMdm, MMbh, MMtotal, Mfdm, \
             projMMstar, projMMdm, projMMtotal, projMfdm, \
                Re_model = _dyLens_Lens_analysis(CM=CM, R=R, dyLens=dyLens)
        
        print("Model Einstein ring: {:.2e} Msun".format( float(Re_model) ))
        print("Data  Einstein ring: {:.2e} Msun".format( float(Re_data) ))
        DthetaE = float ( (Re_data - Re_model)/Re_data )
        print("(Model - Data)/Data: {:.2f}".format( float(DthetaE) ))
        print('=' * term_size.columns)
        

        # Data quantities
    info = fits.open(data_path+"/imgs/log_img.fits")[1].data

    # Load the snapshot data
    dm_dataset   = np.load(data_path+"/dm/coordinates_dark.npy")
    star_dataset = np.load(data_path+"/imgs/coordinates_star.npy")
        
    Mstar, Mdm, Mbh, Mtotal, fdm = quantities3D(star_dataset=star_dataset,
                                                    dm_dataset=dm_dataset, 
                                                    info=info, R_kpc=R_kpc) # 3D quantities

    projMstar, projMdm, Mbh, projMtotal, projfdm = quantities2D(star_dataset=star_dataset,
                                                                    dm_dataset=dm_dataset, 
                                                                    info=info, R_kpc=R_kpc) # 2D quantities

    print("3D results")
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

    print("\n\n\n 2D results")
    print('=' * term_size.columns)
        # Accuracy in proj stellar mass
    print("Model proj stellar Mass: {:.2e} Msun".format( float(projMMstar) ))
    print("Data  proj stellar Mass: {:.2e} Msun".format( float(projMstar) ))
    projDstar = float ( (projMMstar - projMstar)/projMstar )
    print("(Model - Data)/Data: {:.2f}".format( float(projDstar) ))
    print('=' * term_size.columns)
        # Accuracy in Dm mass
    print("Model proj dm Mass: {:.2e} Msun".format( float(projMMdm) ))
    print("Data  proj dm Mass: {:.2e} Msun".format( float(projMdm) ))
    projDdm = float ( (projMMdm - projMdm)/projMdm )
    print("(Model - Data)/Data: {:.2f}".format( float(projDdm) ))
    print('=' * term_size.columns)
        # Accuracy in total mass
    print("Model proj total Mass: {:.2e} Msun".format( float(projMMtotal) ))
    print("Data  proj total Mass: {:.2e} Msun".format( float(projMtotal) ))
    projDtotal = float ( (projMMtotal - projMtotal)/projMtotal )
    print("(Model - Data)/Data: {:.2f}".format( float(projDtotal) ))
    print('=' * term_size.columns)
        # Accuracy in dm fraction
    print("Model proj DM fraction: {:.2f}".format( float(projMfdm) ))
    print("Data  proj DM fraction: {:.2f}".format( float(projfdm) ))
    projDfdm = float ( (projMfdm - projfdm)/projfdm )
    print("(Model - Data)/Data: {:.2f}".format( float(projDfdm) ))

    if JAM:
        _3D_density_profile(Jam_Model=Jam_Model, reff=reff, 
                            data_path=data_path, analysis_path=analysis_path)
    elif dyLens:
        _3D_density_profile(Jam_Model=CM.Jampy_model, reff=reff, 
                            data_path=data_path, analysis_path=analysis_path)
    else: pass

    # Save quantities
    r = {}
    r["Reff"]   = reff
    r["R"]      = float( R )
    r["MthetaE"]= float( Re_data )
    r["Mstar"]  = float( np.log10(Mstar) )
    r["Mdm"]    = float( np.log10(Mdm)   )
    r["Mbh"]    = Mbh
    r["Mtotal"] = float( np.log10(Mtotal) )
    r["fdm"]    = fdm
    r["projMstar"]  = float( np.log10(projMstar) )
    r["projMdm"]    = float( np.log10(projMdm)   )
    r["projMtotal"] = float( np.log10(projMtotal) )
    r["projfdm"]    = projfdm
    r["MMstar"]  = float( np.log10(MMstar) )
    r["MMdm"]    = float( np.log10(MMdm)   )
    r["MMbh"]    = float( np.log10(MMbh)   )
    r["MMtotal"] = float( np.log10(MMtotal) )
    r["Mfdm"]    = float( Mfdm )
    r["MMthetaE"]= float (Re_model)
    r["projMMstar"]  = float( np.log10(projMMstar) )
    r["projMMdm"]    = float( np.log10(projMMdm)   )
    r["projMMtotal"] = float( np.log10(projMMtotal) )
    r["projMfdm"]    = float( projMfdm )
    r["Dstar"]   = Dstar
    r["Ddm"]     = Ddm
    r["Dtotal"]  = Dtotal
    r["Dfdm"]    = Dfdm
    r["projDstar"]   = projDstar
    r["projDdm"]     = projDdm
    r["projDtotal"]  = projDtotal
    r["projDfdm"]    = projDfdm
    r["DthetaE"]     = DthetaE 

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
       
    parser.add_option('--JAM', action='store_true', dest='JAM',
                    default=False, 
                    help='Flag to analyse JAM results')
    parser.add_option('--dyLens', action='store_true', dest='dyLens',
                    default=False, 
                    help='Flag to analyse dyLens results.')
    parser.add_option('--Lens', action='store_true', dest='Lens',
                    default=False, 
                    help='Flag to analyse Lens results.')
    parser.add_option('--phase', action='store', type=str, dest='phase',
                    default='None', 
                    help='Phase to be analysed.')
    parser.add_option('--Re', action='store_true', dest='Re',
                      default=False, 
                      help='Flag to analyse results inside the Einstein ring.')
    
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

   
    run(result_path=result_path, data_path=data_path, 
            JAM=options.JAM, dyLens=options.dyLens, Lens=options.Lens,
            alpha=options.alpha, Re=options.Re, phase_name=options.phase)
    sys.exit()
