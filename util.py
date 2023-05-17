"""
Author: Carlos Melo <carlos.melo@ufrgs.br>
Date: 17/05/2023
Last Mofidication by: Carlos Melo <carlos.melo@ufrgs.br>
Date of last modification: 17/05/2023

Some useful utilities.
"""

import numpy as np

def quantities3D(star_dataset, dm_dataset, info, R_kpc):
    """
    Computes 3D quantities given a dataset (stellar or dm) of particle data.
    All quantities are calculated within a sphere of radius R_kpc.
    This assumes x,y are in the plane of the sky.
    
    Inputs:
    ------------
    star_dataset: Illustris dataset
        The stellar dataset as described in my_illustris/make_img.py
    dm_dataset: Illustris dataset
        The stellar dataset as described in my_illustris/make_img.py
    info: info dict
        Info dict as descibed in my_illustris/make_img.py
    R_kpc[kpc]: float
        Radius, in kpc, of the sphere
    
    Output:
    ------------
    Mstar[Msun]: float
        Stellar mass within R
    Mdm[Msun]: float
        Dark matter mass within R
    Mbh[Msun]: float
        BH mass
    Mtotal[Msun]: float
        Total mass within R
    fdm: float
        dark matter fraction within R
    """

     # Stellar content
    rStar = np.sqrt(np.sum(star_dataset[:, 0:3]**2, axis=1)) # radius
    i = rStar <= R_kpc                                       # only particles within R
    Mstar = sum(star_dataset[:, 6][i]*1e10)                  # mass within R

        # Dark content
    rDM = np.sqrt(np.sum(dm_dataset[:, 0:3]**2, axis=1))
    i = rDM <= R_kpc
    Mdm = sum(dm_dataset[:,3][i]*1e10)
    if info["logMbh"].size != 1:                 # If there is more than one BH
        Mbh = list(np.float_(info["logMbh"][0])) # Sould be the BH mass from TNG catalogue 
    else:
        Mbh = float( info["logMbh"] ) # Sould be the BH mass from TNG catalogue 
    Mtotal = Mstar + Mdm              # Total snapshot mass within R
    fdm    = Mdm/Mtotal               # DM fraction in the snapshot within R

    return Mstar, Mdm, Mbh, Mtotal, fdm

def quantities2D(star_dataset, dm_dataset, info, R_kpc):
    """
    Computes 2D quantities given a dataset (stellar or dm) of particle data.
    All quantities are calculated within a circle of radius R_kpc.
    This assumes x,y are in the plane of the sky.

    Inputs:
    ------------
    star_dataset: Illustris dataset
        The stellar dataset as described in my_illustris/make_img.py
    dm_dataset: Illustris dataset
        The stellar dataset as described in my_illustris/make_img.py
    info: info dict
        Info dict as descibed in my_illustris/make_img.py
    R_kpc[kpc]: float
        Radius, in kpc, of the circle
    
    Output:
    ------------
    Mstar[Msun]: float
        Stellar mass within R
    Mdm[Msun]: float
        Dark matter mass within R
    Mbh[Msun]: float
        BH mass
    Mtotal[Msun]: float
        Total mass within R
    fdm: float
        dark matter fraction within R
    """

     # Stellar content
    rStar = np.sqrt(np.sum(star_dataset[:, 0:2]**2, axis=1)) # radius
    i = rStar <= R_kpc                                       # only particles within R
    Mstar = sum(star_dataset[:, 6][i]*1e10)                  # mass within R

        # Dark content
    rDM = np.sqrt(np.sum(dm_dataset[:, 0:2]**2, axis=1))
    i = rDM <= R_kpc
    Mdm = sum(dm_dataset[:,3][i]*1e10)
    if info["logMbh"].size != 1:                 # If there is more than one BH
        Mbh = list(np.float_(info["logMbh"][0])) # Sould be the BH mass from TNG catalogue 
    else:
        Mbh = float( info["logMbh"] ) # Sould be the BH mass from TNG catalogue 
    Mtotal = Mstar + Mdm              # Total snapshot mass within R
    fdm    = Mdm/Mtotal               # DM fraction in the snapshot within R

    return Mstar, Mdm, Mbh, Mtotal, fdm