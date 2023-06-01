# Performs the analysis of results after the non-linear search.
# Several integrated quantities are measured, as well
# as their comparison with the reference values.


import sys
from optparse import OptionParser

from dynesty import plotting as dyplot
from dynesty import utils as dyfunc


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats.mstats import mquantiles

plt.rcParams['xtick.labelsize']= 12
plt.rcParams['ytick.labelsize']= 12

import pickle
import os
import json

from jampy.mge_half_light_isophote import mge_half_light_isophote
from astropy import units as u


from util import quantities2D, quantities3D
from dyLens.utils.tools import effective_einstein_radius_from_kappa
from dyLens.utils import Analysis

def _plot_density(true_density, model_density,
                lo, hi, r, reff, rmax, thetaE,
                label_true, label_model, label_band, 
                 fit=None, label_fit=None, 
                 save_path=None, save_name=None):
    
    fig1 = plt.figure(figsize=(15,6))

        #Plot Data-model
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    plt.plot(r, true_density,  ".", label=label_true, color="black", markersize=10)
    plt.plot(r, model_density, label=label_model, color="magenta")
    if fit is not None:
        plt.plot(r, fit, label=label_fit, linewidth=2, color="darkblue")
    plt.fill_between(r, lo, hi, color="gray", alpha=0.6, label=label_band)
    plt.axvline(reff, label="$R_{eff}$", linestyle="--", markersize=12, color="navy")
    plt.axvline(rmax, label="Max. kin. data", linestyle="-.", markersize=12, color="navy")
    if thetaE == 0:
        pass
    else:
        plt.axvline(thetaE, label=r"$\theta^{ \rm true }_{ \rm E}$", linestyle=":", markersize=12, color="navy")

    plt.ylabel("$\\rho \,[M_{\odot}/pc^3]$", size=20)
    plt.loglog()
    plt.legend(ncol=2)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame

        #Residual plot
    frame2   = fig1.add_axes((.1,.1,.8,.2))       
    residual = ( true_density  - model_density ) / true_density
    residual_lo = ( true_density  - lo ) / true_density
    residual_hi = ( true_density  - hi ) / true_density
    plt.plot(r, 100*residual, 'd', color='LimeGreen', mec='LimeGreen', ms=4)
    plt.fill_between(r, 100*residual_lo, 100*residual_hi, color="gray", alpha=0.25)
    plt.grid(axis = 'y', linestyle="--")

    plt.xlabel("radii [pc]", size=20)
    plt.ylabel("Residual (%)", size=12)
    plt.xscale("log")

    if save_path:
        plt.savefig(save_path+"/{}.png".format(save_name))
        plt.close()

def run(result_path, data_path, JAM=None, dyLens=None, Lens=None,
            alpha=None, Re=None, phase_name=None, ncores=8):
    
    if JAM == dyLens == Lens == None:
        raise ValueError("You must provide at least one of the keywords related to the analysis: JAM, Lens or Lens")
    else:
        if (JAM and Lens) or (JAM and dyLens) or (Lens and dyLens):
            raise ValueError("You must provide only one keyword to be analysed.")
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
            try:
                sampler = sampler["sampler"]
            except:
                sampler = sampler

        with open(result_path+f'/JAM_class.pickle','rb') as f:
            Jam_Model = pickle.load(f)
            f.close()

        with open(result_path+f'/priors.pickle','rb') as f:
            priors = pickle.load(f)
            f.close()
            
            # An object to analyse the results
        A = Analysis.Analysis(sampler=sampler, 
                                Jampy_model=Jam_Model, ncores=ncores)
    
    elif dyLens:
            # Load sampler, phase and model of dyLens
        with open(result_path+'/Final_sampler_{}.pickle'.format(phase_name),'rb') as f:
            sampler = pickle.load(f)
            f.close()
            try:
                sampler = sampler["sampler"]
            except:
                sampler = sampler

        with open(result_path+'/{}.pickle'.format(phase_name),'rb') as f:
            phase = pickle.load(f)
            f.close()

        with open(result_path+'/CombinedModel_{}.pickle'.format(phase_name),'rb') as f:
            CM = pickle.load(f)
            f.close()
            
        with open(result_path+'/priors_{}.pickle'.format(phase_name),'rb') as f:
            priors = pickle.load(f)
            f.close()
        
            # An object to analyse the results
        A = Analysis.Analysis(sampler=sampler, 
                                dyLens_phase=phase, ncores=ncores)
        
    elif Lens:
            # Load sampler, phase and model of Lens
        with open(result_path+'/Final_sampler_{}.pickle'.format(phase_name),'rb') as f:
            sampler = pickle.load(f)
            f.close()
            try:
                sampler = sampler["sampler"]
            except:
                sampler = sampler

        with open(result_path+'/{}.pickle'.format(phase_name),'rb') as f:
            phase = pickle.load(f)
            f.close()

            # An object to analyse the results
        A = Analysis.Analysis(sampler=sampler, 
                                Lens_phase=phase, ncores=ncores)

    

        # Tracer plot
    A.tracerPlot(save_path=analysis_path)
    plt.close()
    
        # Plot the 2-D marginalized posteriors.
    A.cornerPlot(save_path=analysis_path)
    plt.close()
    
    if dyLens:
        A.DynFiducial(save_path=analysis_path)
        plt.close()
        A.LensFiducial(save_path=analysis_path)
        plt.close()
    elif JAM:
        A.DynFiducial(save_path=analysis_path)
        plt.close()
    else:
        A.LensFiducial(save_path=analysis_path)
        plt.close()
       
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
                    "MMstar": "Model stellar mass and 68% CL within R, in log10.", 
                    "MMdm":   "Model dm mass and 68% CL within R, in log10.", 
                    "MMbh":   "Model BH mass within R, in log10.", 
                    "MMtotal":"Model total mass and 68% CL within R, in log10.", 
                    "Mfdm":   "Model DM fraction and 68% CL within R.",
                    "MMthetaE": "Measured Einstein Ring in arcsec",
                    "projMMstar": "Projected model stellar mass and 68% CL within R, in log10.", 
                    "projMMdm":   "Projected model dm mass and 68% CL within R, in log10.", 
                    "projMMtotal":"Projected model total mass and 68% CL within R, in log10.", 
                    "projMfdm":   "Projected model DM fraction and 68% CL within R.",
                    "Dstar":  "(MMstar - Mstar)/Mstar with respect the median",
                    "Ddm":    "(MMdm - Mdm)/Mdm with respect the median",
                    "Dtotal": "(MMtotal - Mtotal)/Mtotal with respect the median",
                    "Dfdm":   "(Mfdm - fdm)/fdm with respect the median",
                    "projDstar":  "proj (MMstar - Mstar)/Mstar with respect the median",
                    "projDdm":    "proj (MMdm - Mdm)/Mdm with respect the median",
                    "projDtotal": "proj (MMtotal - Mtotal)/Mtotal with respect the median",
                    "projDfdm":   "proj (Mfdm - fdm)/fdm with respect the median",
                    "DthetaE": "(MthetaE - MMthetaE)/MthetaE"
                }
    json.dump(description, out_descripition, indent = 8)
    out_descripition.close()
    
        # Get the effective radius in arcsec, and other quantities
        # See mge_half_light_isophote documentation for more details
    if JAM:       
        distance = Jam_Model.distance # Distance in Mpc
        reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(
                                                Jam_Model.surf_lum,
                                                Jam_Model.sigma_lum,
                                                Jam_Model.qobs_lum,
                                                distance
                                                )
        Rmax_data = max( np.sqrt( Jam_Model.xbin**2 + Jam_Model.ybin**2 ) )
    else: 
        distance = A.phase.CombinedModel.cosmology.angular_diameter_distance(
                                            z=A.phase.CombinedModel.Lens_model.z_l).value # Distance in Mpc
        reff, reff_maj, eps_e, lum_tot = mge_half_light_isophote(
                                                    A.phase.CombinedModel.Lens_model.surf_lum,
                                                    A.phase.CombinedModel.Lens_model.sigma_lum,
                                                    A.phase.CombinedModel.Lens_model.qobs_lum,
                                                    distance
                                                    )
        if dyLens:
            Rmax_data = max( np.sqrt( A.phase.CombinedModel.Jampy_model.xbin**2 + 
                                      A.phase.CombinedModel.Jampy_model.ybin**2 ) )
        else:
            Rmax_data = None
    
        # True Einstein ring 
    if dyLens or Lens:
            # surface mass density in 1e10 Msun/kpc2
        sigma_M = fits.open(data_path+"/imgs/surface_mass_density.fits")[0]        
            # kappa from data. The critical density is in Msun/Mpc2, so I convert it to Msun/kpc2
        kappa_data = (sigma_M.data*1e10)/ ( A.phase.CombinedModel.Lens_model.critical_density/(1e3)**2 )
        Re_data  = effective_einstein_radius_from_kappa(
                                    kappa_data, 
                                    grid_spacing=A.phase.CombinedModel.Fit.image.pixel_scale,
                                    grid=A.phase.CombinedModel.Fit.imaging.unmasked.grid)
    else: pass

    if Re:
        R = Re_data     # Einstein ring in arcsec
    else: 
        R = alpha*reff  # alpha times the Reff in arcsec
    R_kpc = ( (R*u.arcsec * distance*u.Mpc ).to(
                        u.kpc,u.dimensionless_angles()) ).value # value in kpc

        # Integrated quantities
    if JAM or dyLens:
            # 3D Quantities
        MMstar  = A.stellarMass_3D(R=R, save_path=analysis_path)
        MMdm    = A.dmMass_3D(R=R, save_path=analysis_path)
        MMbh    = 0.0       #TODO: Fix it in a new version of Analysis class

        MMTotal_dist = np.asarray( A.stellar_mass3D_dist) + \
                        np.asarray( A.dm_mass3D_dist)            # Total mass distribution
        MMtotal   = dyfunc.quantile(MMTotal_dist, 
                                    q=[0.16, 0.5, 0.84],
                                    weights=A.weights)
        
        Mfdm_dist  = np.asarray(A.dm_mass3D_dist) / MMTotal_dist # DM fraction distribution
        Mfdm       = dyfunc.quantile(Mfdm_dist, 
                                    q=[0.16, 0.5, 0.84],
                                    weights=A.weights)
            # 2D Quantities
        projMMstar  = A.stellarMass_2D(R=R, save_path=analysis_path)
        projMMdm    = A.dmMass_2D(R=R, save_path=analysis_path)

        projMtotal_dist = np.asarray( A.stellar_mass2D_dist ) + \
                            np.asarray( A.dm_mass2D_dist )       # Total proj mass distribution
        projMMtotal =  dyfunc.quantile(projMtotal_dist, 
                                    q=[0.16, 0.5, 0.84],
                                    weights=A.weights)
        
        projMfdm_dist = np.asarray(A.dm_mass2D_dist) / projMtotal_dist # proj DM fraction distribution
        projMfdm      = dyfunc.quantile(projMfdm_dist, 
                                    q=[0.16, 0.5, 0.84],
                                    weights=A.weights)

    else:
            # 3D Quantities
        MMstar = MMdm = MMtotal = Mfdm = [0.0, 0.0, 0.0]  # Lens only is not sensitive to them.
        MMbh = 0.0
            # 2D Quantities
        projMMstar  = A.stellarMass_2D(R=R, save_path=analysis_path)
        projMMdm    = A.dmMass_2D(R=R, save_path=analysis_path)

        projMtotal_dist = np.asarray( A.stellar_mass2D_dist ) + \
                            np.asarray( A.dm_mass2D_dist )       # Total proj mass distribution
        projMMtotal =  dyfunc.quantile(projMtotal_dist, 
                                    q=[0.16, 0.5, 0.84],
                                    weights=A.weights)
        
        projMfdm_dist = np.asarray(A.dm_mass2D_dist) / projMtotal_dist # proj DM fraction distribution
        projMfdm      = dyfunc.quantile(projMfdm_dist, 
                                    q=[0.16, 0.5, 0.84],
                                    weights=A.weights)

        # Measured Einstein Ring
    if JAM:
            # JAM cannot constraint it
        Re_model = DthetaE = Re_data =0
    else:
            # kappa MGE model. Uses the same grid as the lens image
        kappa_model = A.phase.CombinedModel.Lens_model.convergence_2d_from(
                        A.phase.CombinedModel.Fit.imaging.unmasked.grid)

        # Calculates model the Einstein ring
        # Uses the same grid as the lens image
        Re_model = effective_einstein_radius_from_kappa(
                            kappa_model, 
                            grid_spacing=A.phase.CombinedModel.Fit.image.pixel_scale,
                            grid=A.phase.CombinedModel.Fit.imaging.unmasked.grid, 
                            Nsamples=A.phase.CombinedModel.Fit.imaging.unmasked.grid.shape[0]/2)
        print("\n")
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
        
    Mstar, Mdm, Mbh, Mtotal, fdm = quantities3D(
                                            star_dataset=star_dataset,
                                            dm_dataset=dm_dataset, 
                                            info=info, R_kpc=R_kpc)             # 3D quantities

    projMstar, projMdm, Mbh, projMtotal, projfdm = quantities2D(
                                                        star_dataset=star_dataset,
                                                        dm_dataset=dm_dataset, 
                                                        info=info, R_kpc=R_kpc) # 2D quantities

    print("3D results")
    print('=' * term_size.columns)
        # Accuracy in stellar mass
    print("Model stellar Mass: {:.2e} Msun".format( float(MMstar[1]) ))
    print("Data  stellar Mass: {:.2e} Msun".format( float(Mstar) ))
    Dstar = float ( (MMstar[1] - Mstar)/Mstar )
    print("(Model - Data)/Data: {:.2f}".format( float(Dstar) ))
    print('=' * term_size.columns)
        # Accuracy in Dm mass
    print("Model dm Mass: {:.2e} Msun".format( float(MMdm[1]) ))
    print("Data  dm Mass: {:.2e} Msun".format( float(Mdm) ))
    Ddm = float ( (MMdm[1] - Mdm)/Mdm )
    print("(Model - Data)/Data: {:.2f}".format( float(Ddm) ))
    print('=' * term_size.columns)
        # Accuracy in total mass
    print("Model total Mass: {:.2e} Msun".format( float(MMtotal[1]) ))
    print("Data  total Mass: {:.2e} Msun".format( float(Mtotal) ))
    Dtotal = float ( (MMtotal[1] - Mtotal)/Mtotal )
    print("(Model - Data)/Data: {:.2f}".format( float(Dtotal) ))
    print('=' * term_size.columns)
        # Accuracy in dm fraction
    print("Model DM fraction: {:.2f}".format( float(Mfdm[1]) ))
    print("Data  DM fraction: {:.2f}".format( float(fdm) ))
    Dfdm = float ( (Mfdm[1] - fdm)/fdm )
    print("(Model - Data)/Data: {:.2f}".format( float(Dfdm) ))

    print("\n\n\n 2D results")
    print('=' * term_size.columns)
        # Accuracy in proj stellar mass
    print("Model proj stellar Mass: {:.2e} Msun".format( float(projMMstar[1]) ))
    print("Data  proj stellar Mass: {:.2e} Msun".format( float(projMstar) ))
    projDstar = float ( (projMMstar[1] - projMstar)/projMstar )
    print("(Model - Data)/Data: {:.2f}".format( float(projDstar) ))
    print('=' * term_size.columns)
        # Accuracy in Dm mass
    print("Model proj dm Mass: {:.2e} Msun".format( float(projMMdm[1]) ))
    print("Data  proj dm Mass: {:.2e} Msun".format( float(projMdm) ))
    projDdm = float ( (projMMdm[1] - projMdm)/projMdm )
    print("(Model - Data)/Data: {:.2f}".format( float(projDdm) ))
    print('=' * term_size.columns)
        # Accuracy in total mass
    print("Model proj total Mass: {:.2e} Msun".format( float(projMMtotal[1]) ))
    print("Data  proj total Mass: {:.2e} Msun".format( float(projMtotal) ))
    projDtotal = float ( (projMMtotal[1] - projMtotal)/projMtotal )
    print("(Model - Data)/Data: {:.2f}".format( float(projDtotal) ))
    print('=' * term_size.columns)
        # Accuracy in dm fraction
    print("Model proj DM fraction: {:.2f}".format( float(projMfdm[1]) ))
    print("Data  proj DM fraction: {:.2f}".format( float(projfdm) ))
    projDfdm = float ( (projMfdm[1] - projfdm)/projfdm )
    print("(Model - Data)/Data: {:.2f}".format( float(projDfdm) ))

        # 3D density profiles
    if JAM or dyLens:
        analysis_densities = analysis_path+"/densities/"
        os.makedirs(analysis_densities)
        pc       = distance*np.pi/0.648  # Constant factor to convert arcsec --> pc
        reff_pc  = reff*pc
        Re_data_pc   = Re_data*pc
        Rmax_data_pc = Rmax_data*pc
        
            # Load DM density profile
        dm_hdu = fits.open(data_path+"/dm/density_fit.fits")
        rho_dm = dm_hdu[1].data["density"]
        r_dm   = dm_hdu[1].data["radius"]
        dm_fit = dm_hdu[1].data["bestfit"]
        
        i = r_dm < 2*Rmax_data_pc  # Two times the max kin data
        rho_dm = rho_dm[i]
        r_dm   = r_dm[i]
        dm_fit = dm_fit[i]

            # Load star density profile
        star_hdu = fits.open(data_path+"/imgs/stellar_density.fits")
        rho_star = star_hdu[1].data["density"]
        r_star   = star_hdu[1].data["radius"]

        i = r_star < 2*Rmax_data_pc  # Two times the max kin data
        rho_star = rho_star[i]
        r_star   = r_star[i]
            
            # This assumes that dm and stellar profiles were
            # evaluated at same points
        radii = r_star/pc     # radii where to compute the models, in arsec

        dstar  = np.asarray( A.stellarMass3D_density(radii=radii, 
                                                     save_path=analysis_densities)
                        )
        ddm    = np.asarray( A.dmMass3D_density(radii=radii,
                                                save_path=analysis_densities)
                        )
        dtotal = dstar + ddm
            # median and 1sigma band
        mean_star = mquantiles(dstar, 0.5, axis=0)[0]
        lo_star   = mquantiles(dstar, 0.16, axis=0)[0]
        hi_star   = mquantiles(dstar, 0.84, axis=0)[0]

        mean_dm = mquantiles(ddm, 0.5, axis=0)[0]
        lo_dm   = mquantiles(ddm, 0.16, axis=0)[0]
        hi_dm   = mquantiles(ddm, 0.84, axis=0)[0]

        mean_total = mquantiles(dtotal, 0.5, axis=0)[0]
        lo_total   = mquantiles(dtotal, 0.16, axis=0)[0]
        hi_total   = mquantiles(dtotal, 0.84, axis=0)[0]
        
            # Plot stellar
        _plot_density(rho_star, mean_star, 
                lo_star, hi_star, r_star, reff_pc, Rmax_data_pc, Re_data_pc,
                label_true=r"Stellar$_{\rm true}$",
                label_model=r"Stellar$_{\rm model}$",
                label_band="$1\sigma$",
                save_path=analysis_densities, save_name="stellar_density")

            # Plot dm
        _plot_density(rho_dm, mean_dm, 
                lo_dm, hi_dm, r_dm, reff_pc, Rmax_data_pc, Re_data_pc,
                label_true=r"DM$_{\rm true}$",
                label_model=r"DM$_{\rm model}$",
                label_band="$1\sigma$",
                fit=dm_fit, label_fit=r"DM$_{\rm Fit}$",
                save_path=analysis_densities, save_name="dm_density")

            # Plots total
        _plot_density(rho_star + rho_dm, mean_total, 
                lo_total, hi_total, r_dm, reff_pc, Rmax_data_pc, Re_data_pc,
                label_true=r"Total$_{\rm true}$",
                label_model=r"Total$_{\rm model}$",
                label_band="$1\sigma$",
                save_path=analysis_densities, save_name="total_density")

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
    r["projMstar"]  = float( np.log10(projMstar)  )
    r["projMdm"]    = float( np.log10(projMdm)    )
    r["projMtotal"] = float( np.log10(projMtotal) )
    r["projfdm"]    = projfdm
    r["MMstar"]  = list( np.log10(MMstar)  )
    r["MMdm"]    = list( np.log10(MMdm)    )
    r["MMbh"]    = float( np.log10(MMbh)   )
    r["MMtotal"] = list( np.log10(MMtotal) )
    r["Mfdm"]    = list( Mfdm )
    r["MMthetaE"]= float (Re_model)
    r["projMMstar"]  = list( np.log10(projMMstar)  )
    r["projMMdm"]    = list( np.log10(projMMdm)    )
    r["projMMtotal"] = list( np.log10(projMMtotal) )
    r["projMfdm"]    = list( projMfdm )
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

    if options.JAM and options.Re:
        raise ValueError("Jampy model is not able to infer the Einstein radius.")


    result_path = args[0] # Path to the results folder
    data_path   = args[1] # Path to the data folder

   
    run(result_path=result_path, data_path=data_path, 
            JAM=options.JAM, dyLens=options.dyLens, Lens=options.Lens,
            alpha=options.alpha, Re=options.Re, phase_name=options.phase)
    sys.exit()
