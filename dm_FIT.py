import sys
import gc
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15
from scipy.optimize import curve_fit
from dynesty import NestedSampler     #sampler
from dynesty import plotting as dyplot
from dynesty.utils import quantile
from dynesty import utils as dyfunc
import dynesty.pool as dypool
from astropy.io import fits
from numba import jit
import json



@jit(nopython=True)
def gNFW(r, rho_s, r_s, inner_slope):
    return rho_s * (r/r_s)**(-inner_slope) * (1 + r/r_s)**(inner_slope - 3)

@jit(nopython=True)
def NFW(r, rho_s, r_s):
    return  (rho_s * r_s) / (r * (1 + r/r_s)**2 )

@jit(nopython=True)
def Burkert(r, rho_c, r_c):
    """
        Dark matter profile with a core radius.
        See https://arxiv.org/pdf/2303.12859.pdf
    """
    c = r/r_c
    return rho_c / ( (1 + c**2) * (1 + c) )

@jit(nopython=True)
def Einasto(r, rho_0, r_0, alpha):
    """
        See https://doi.org/10.3390/galaxies8040074
    """
    return rho_0 * np.exp( 2.0 * alpha * ( 1 - (r/r_0)**(1/alpha) )  )


    #Dic with available models.
    #The list contains the number of free parameters and the function to be fitted.
models_list ={
                "NFW": [2, NFW], 
                "gNFW": [3, gNFW],
                "Burkert": [2, Burkert],
                "Einasto": [3, Einasto]
                }  

model_labels = {
                "NFW":  [r'$\rho_s$', r'$r_s$'],
                "gNFW": [r'$\rho_s$', r'$r_s$', r'$\gamma$'],
                "Burkert":  [r'$\rho_c$', r'$r_c$'],
                "Einasto": [r'$\rho_0$', r'$r_0$', r'$\alpha$'],
                }



model_parameters = {
                "NFW":  ['rho_s', 'r_s'],
                "gNFW": ['rho_s', 'r_s', 'gamma'],
                "Burkert": ['rho_c', 'r_c'],
                "Einasto": ['rho_0', 'r_0', 'alpha'],
                }
#In this case, the parameters are:
        #rho_s       - units of Msun/pc3
        #r_s         - units of pc
        #inner slope - no units
        #rho_c       - units of Msun/pc3
        #r_c         - units of pc
        #rho_0       - units ofMsun/pc3
        #r_0         - units of pc
        #alpha       - no units
model_bounds = {
                "NFW":  ([0, 1], [1e5, np.inf]),
                "gNFW": ([0, 0.1, 0], [1e5, np.inf, 2]),  
                "Burkert":  ([0, 0.1], [1e5, np.inf]),
                "Einasto": ([0, 0.1, 0], [1e5, np.inf, 12.]),  
                }               
##########################################################################
# Here we will define some some configurations for the non-linear search 
# In this case, the parameters are:
        #log10(rho_s)  - units of log10(Msun/pc3)
        #r_s           - units of arcsec
        #inner slope   - no units
        #log10(rho_c)  - units of log10(Msun/pc3)
        #log10(rho_0)  - units of log10(Msun/pc3)
        #r_c           - units of arcsec
        #r_o           - units of arcsec
        #alpha         - no units
non_linear_bounds = {
                "NFW":  (np.array([-6, 1e-2]), np.array([0, 30])),
                "gNFW": (np.array([-6, 1e-2, 0]), np.array([0, 30, 2])), 
                "Burkert":  (np.array([-6, 1e-2]), np.array([0, 30])), 
                "Einasto": (np.array([-6, 1e-2, 0.]), np.array([0, 30, 12.])), 
                }
    # Labels for non-linear fit
model_labels_nonlinear = {
                "NFW":  [r'$\log_{10}(\rho_s)$', r'$r_s$'],
                "gNFW": [r'$\log_{10}(\rho_s)$', r'$r_s$', r'$\gamma$'],
                "Burkert":  [r'$\log_{10}(\rho_c)$', r'$r_c$'],
                "Einasto": [r'$\log_{10}(\rho_0)$', r'$r_0$', r'$\alpha$'],
                }
    # Sampler configuration
    # To change the sampler config, change it here!
_sampler_config = {
                    "nlive" : 180,
                    "sample": 'rwalk', 
                    "walks" : 15,
                    "bound" : 'multi',
                    }


class Profile_fit(object):
    def __init__(self, density, radius, z, path,
                     sigma=None, cosmology=Planck15,
                     model="NFW", fraction=None) -> None:
        """
        A class to fit DM spherical density profiles using different models.
        For now, the available models are:
        NFW:
            (rho_s * r_s) / (r * (1 + r/r_s)**2 )
        gNFW:
            rho_s * (r/r_s)**(-inner_slope) * (1 + r/r_s)**(inner_slope - 3)
        Burkert:
            rho_c / ( (1 + c**2) * (1 + c) )
            with c = r/r_c
        Einasto:
           rho_0 * np.exp( 2.0 * alpha * (1 - r/r_0)**(1/alpha)  )

        Parameters are:
        rho_s/rho_c/rho_0
            characteristic density at the scale radius/at the core radius
        r_s/r_c/r_0
            characteristic radius/core radius
        slope
            inner density slope
        alpha
            degree of curvature (shape) of the Einasto profile.
            If alpha > 4 are identified with cuspy halos, while 
            for alpha < 4 presents a cored-like behavior. 
        r
            radius where to compute the density profile

        Inputs:
        ----------
            density: 1D-array
                Density to be fitted using a DM profile.
                Supposed to be in M_sun/pc**3. However, the bounds 
                of the parameters can be changed in fit(), so that the 
                density units can be re-defined.
            radius: 1D-array
                Radius where the density was computed. 
                Supposed to be in pc.
            z: float
                redshift of the halo.
            path: str
                path where the data will be save
            sigma: 1D-array
                Error on the density profile. If none are 
                defined, the error is assumed to be a fraction
                of the density input. See  fraction bellow.
            model: str
                Model to be fitted to the data. 
                The models currently available are:
                    NFW
                    gNFW
            cosmology: astropy cosmology
                Astropy cosmology. Default is Planck15.
                Used to transform arcsec to pc. 
            fraction: float [0., 1.]
                Fraction to be used as a proxy for the error
                in the density profile. Default is 15%,
                so the error will be just 0.15 * density.
        """

        assert density.size == radius.size, "density and radius should be the same size."

        if sigma != None:
            assert density.size == sigma.size, "error and density should be the same size."
        else:
            if fraction != None:
                assert 0.< fraction < 1.0, "fraction should be between 0. and 1."
                sigma = density * fraction
            else:
                sigma = density * 0.15
        
        assert model in models_list.keys(),  "Set model is not available."

        self.density = density
        self.radius  = radius
        self.sigma   = sigma
        self.model   = model
        self.path    = path
        self.D       = cosmology.angular_diameter_distance(z) #angular diameter distance
        pass

    def fit(self, bounds=None, maxfev=5000, plot=False, **kwargs):
        """
        Fit the density to the model.
        The results are stored as attributes of the class:
            popt: Optimal values for the parameters so that the sum of the squared 
                    residuals of ``f(xdata, *popt) - ydata`` is minimized.
            pcov: The estimated covariance of popt.
            perr: Standard deviation errors
            bestfit: best fit density profile
        Optional:
        ----------
            bounds: 2D-tuple
                tuple with the bounds of the parameters.
                Currently, we aply the following bounds:
                    NFW/Burkert:
                        r_s/r_c     - [0, np.inf]   (suppose to be in pc)
                        rho_s/rho_c - [0., 0.5])    (suppose to be in Msun/pc3)
                    gNFW:
                        r_s   - [0, np.inf]   (suppose to be in pc)
                        rho_s - [0., 0.5]     (suppose to be in Msun/pc3)
                        inner_slope - [0, 2]
            
                However, you can use this parameter to set different units, e.g, 
                if your density profile is in units of M_sun/kpc**3, 
                you can set the bounds of r_s to be considered in Kpc 
                instead, reducing the range of the parameter.
            maxfev: int
                Max number of function evaluations
            plot: bool
                Plot the resuts
            **kwargs: any scipy.curve_fit arg.
        """

        maxfev = int(maxfev)
        if bounds != None:
            assert len(bounds[0]) == models_list[self.model][0], "Number of parameters in bounds is \
                        greater than the number of parameters in the set model."
            
            self.popt, self.pcov = curve_fit(models_list[self.model][1], 
                                    self.radius, self.density,  sigma=self.sigma, 
                                    bounds=bounds, maxfev=maxfev, **kwargs)

            self.perr = np.sqrt(np.diag(self.pcov))  #standard deviation errors
        else:
            self.popt, self.pcov = curve_fit(models_list[self.model][1], 
                                    self.radius, self.density,  sigma=self.sigma, 
                                    bounds=model_bounds[self.model],  maxfev=maxfev, **kwargs)

            self.perr = np.sqrt(np.diag(self.pcov))  #standard deviation errors
        
        self.bestfit = models_list[self.model][1](self.radius, *self.popt)
        self.err = 1 - self.bestfit/self.density     #relative error: bestfit, density are positive quantities
        self.absdev = np.mean(np.abs(self.err))      #Mean absolute deviation between fit and data
        
        if plot:
            self._plot()
        else: pass

    def _plot(self, save=False):
        textstr = '\n'.join(
            ["{} = {:.2e} +/- {:.2e}".format(key, self.popt[i], self.perr[i]) for i, key in enumerate(model_labels[self.model])])


        fig, ax = plt.subplots(figsize=(12,8))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        
        ax.plot(self.radius, np.log10(self.density), '.r', label="data")
        ax.plot(self.radius, np.log10(self.bestfit), "b",  label="bestfit")
        ax.set_xlabel("radius")
        ax.set_ylabel("log10(density)")
        ax.set_title(self.model)

        ax.text(0.1, 0.3, textstr, 
            transform=ax.transAxes, 
            bbox=props,
         verticalalignment='top')

        plt.legend()
        if save:
            fig.savefig('{}/dm_fit.png'.format(self.path), dpi=300)
        else:
            plt.show()
    
    def _prior_transform(self, u):
        """
        Transforms unit cube to physical cube.
        Inputs:
        ---------
        u: ndarray
            parameters in the unit cube.
        
        Output:
        ---------
        x: ndarray
            parameters in physical units
        """
        return ( non_linear_bounds[self.model][0] + 
                        u*(non_linear_bounds[self.model][1] - non_linear_bounds[self.model][0]) )
    
    def _loglike(self, pars):
        """
        Log likelihood of the model.
        Inputs:
        ----------
        pars: ndarray
            physical parameters
        
        Outputs:
        ----------
        loglikelihood of the model
        """
        x = np.array(pars) #copy
        x[0] = 10**x[0]    #convert log10(rho) to rho
        x[1] = ( (x[1]*u.arcsec * self.D).to(u.pc, u.dimensionless_angles()) ).value #Scale radius [pc]

        model    = models_list[self.model][1](self.radius, *x)
        residual = self.density - model
        chi_squared_map = (residual / self.sigma) ** 2.0
        
        return -0.5 * np.sum(chi_squared_map)
    
    def _plot_nonlinear(self, results):
        """
            Plot and output results with 68% CF
        """
        
        results_sim = dyfunc.jitter_run(dyfunc.resample_run(results))  #results after combine uncertainties
        samples_sim, weights = results_sim.samples, results_sim.importance_weights()           
        quantiles = [dyfunc.quantile(samps, [0.16, 0.5,  0.84], weights=weights)
                for samps in samples_sim.T]                        #quantiles for each parameter
        
        results   = {}       #dict with the most probable results
        for i, key in enumerate(model_parameters[self.model]):
            results[key] = tuple(quantiles[i])
        
        # the json file where the output must be stored
        out_file = open("{}/dm_nonlinear_rst.json".format(self.path), "w")
        json.dump(results, out_file, indent = 6)
        out_file.close()

        self.results_nonlinear = results

        fig, axes = dyplot.traceplot(results_sim, 
                             labels=model_labels_nonlinear[self.model],
                             quantiles=[0.16, 0.5, 0.84],
                             title_quantiles=[0.16, 0.5, 0.84],
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', connect=True,
                             connect_highlight=range(5))
        fig.tight_layout()
        fig.savefig('{}/dm_tracer.png'.format(self.path), dpi=300)

        fig, axes = dyplot.cornerplot(results_sim,
                              show_titles=True, 
                              title_kwargs={'y': 1.04}, 
                              quantiles=[0.16, 0.5, 0.84],
                              quantiles_2d=[0.16, 0.5, 0.84],
                              title_quantiles=[0.16, 0.5, 0.84],
                              labels=model_labels_nonlinear[self.model],
                              )
        fig.tight_layout()
        fig.savefig('{}/dm_corner.png'.format(self.path), dpi=300)

    def non_linear(self, dlogz_final=0.8, checkpoint_every=60*10,
                    print_progress=False, ncores=2, **kwargs):
        """
        Run dynesty non-linear fit.
        """

        with dypool.Pool(ncores, 
                            self._loglike, 
                            self._prior_transform) as pool:
            
            sampler = NestedSampler(
                                    pool.loglike, 
                                    pool.prior_transform, 
                                    models_list[self.model][0], 
                                    pool=pool, **_sampler_config,
                                    **kwargs 
                                    )
            sampler.run_nested(
                            dlogz=dlogz_final,
                            checkpoint_file="{}/sampler_dm.pickle".format(self.path),
                            checkpoint_every=checkpoint_every,
                            print_progress=print_progress)
            gc.collect()
            
        sampler.pool.close() #End pool processes
        sampler.pool.join() 
        pool.close()
        pool.join()
        sampler.save("{}/sampler_dm.pickle".format(self.path))

        self._plot_nonlinear(sampler.results)
        print("\nNon linear search has finished.")                

    def dumb_to_fit(self, obj):
        """
        Save fit, best fit parameters and others quantities.
        Inputs:
        ----------
            obj: str
                object name
        """
         
        hdu = fits.PrimaryHDU()
        hdu.header.set("object", obj, comment="Object name")
        hdu.header.set("model", self.model, comment="Dark matter model fitted")
        for i, key in enumerate(model_parameters[self.model]):
            hdu.header.set(key, self.popt[i], comment="Parameter best fit")
            hdu.header.set("u_"+key, self.perr[i], comment="Parameter uncertainty")
        hdu.header.set("meanabs", self.absdev, comment="Mean absolute deviation")

        c1 = fits.Column(name='density', format='D', array=self.density)
        c2 = fits.Column(name='bestfit', format='D', array=self.bestfit)
        c3 = fits.Column(name='radius', format='D', array=self.radius)

        coldefs1 = fits.ColDefs([c1, c2, c3])
        tbhdu1   = fits.BinTableHDU.from_columns(coldefs1)
        hdulist  = fits.HDUList([hdu, tbhdu1])
        hdulist.writeto('{}/density_fit.fits'.format(self.path), overwrite=True)
        self._plot(save=True)







