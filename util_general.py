"""
Original Author: Xiaohan Wu <https://xiaohanzai.github.io/>
Last Mofidication by: Carlos Melo <carlos.melo@ufrgs.br>
Date of last modification: 17/05/2023


Some useful utilities.
"""

import numpy as np
from astropy import units as u
from astropy import constants as const

def xyz2cyl(x, v):
	'''
	Convert from (x,y,z) to (R,phi,z).
	z axis should be the symmetry axis.
	'''
	R = np.sqrt(x[:,0]**2 + x[:,1]**2)
	phi = np.arccos(x[:,0]/R)
	id_phi = np.where(x[:,1]<0)[0]
	phi[id_phi] = 2*np.pi - phi[id_phi]
	
	xcyl = np.zeros((len(x),3))
	xcyl[:,0] = R; xcyl[:,1] = phi; xcyl[:,2] = x[:,2]+0.

	vR = v[:,0]*x[:,0]/xcyl[:,0] + v[:,1]*x[:,1]/xcyl[:,0]
	vphi = -v[:,0]*x[:,1]/xcyl[:,0] + v[:,1]*x[:,0]/xcyl[:,0]
	
	vcyl = np.zeros((len(x),3))
	vcyl[:,0] = vR; vcyl[:,1] = vphi; vcyl[:,2] = v[:,2]+0.

	return xcyl, vcyl

def xyz2sph(x, v):
	'''
	Convert from (x,y,z) to (r,phi,theta).
	'''
	r = np.sqrt(np.sum(x**2, axis=1))
	R = np.sqrt(x[:,0]**2 + x[:,1]**2)
	theta = np.arccos(x[:,2]/r)
	phi = np.arccos(x[:,0]/R)
	id_phi = np.where(x[:,1]<0)[0]
	phi[id_phi] = 2*np.pi - phi[id_phi]

	xsph = np.zeros_like(x)
	xsph[:,0] = r; xsph[:,1] = phi; xsph[:,2] = theta

	vR = v[:,0]*x[:,0]/R + v[:,1]*x[:,1]/R
	vphi = -v[:,0]*x[:,1]/R + v[:,1]*x[:,0]/R
	vr = vR*np.sin(theta) + v[:,2]*np.cos(theta)
	vtheta = -vR*np.cos(theta) + v[:,2]*np.sin(theta)

	vsph = np.zeros_like(v)
	vsph[:,0] = vr; vsph[:,1] = vphi; vsph[:,2] = vtheta

	return xsph, vsph

def rotationMatrix(phi, inc, oblate, pa=None):
    """
        Return the rotation matrix given the angles, in deg, 
        and the orientation shape.
    """


    # if oblate:
    #     phi = phi / 180.0 * np.pi
    #     inc = inc / 180.0 * np.pi  # i =90 => edge-on
    #     M_rot_z = np.array([[np.cos(phi), -np.sin(phi), 0],
    #                         [np.sin(phi), np.cos(phi), 0],
    #                         [0, 0, 1]])
    #     M_rot_x = np.array([[1, 0, 0],
    #                         [0, np.cos(inc), -np.sin(inc)],
    #                         [0, np.sin(inc), np.cos(inc)]])
    #     M_rot = M_rot_x@M_rot_z
    # else:
    #     inc = (90-inc) / 180.0 * np.pi  # i =90 => edge-on
    #     phi = phi / 180.0 * np.pi
    #     M_rot_y = np.array([[np.cos(inc), -np.sin(inc), 0],
    #                         [0, 0, 1],
    #                         [np.sin(inc), np.cos(inc), 0]])
    #     M_rot_x = np.array([[1, 0, 0],
    #           alunos         [0, np.cos(phi), -np.sin(phi)],
    #                         [0, np.sin(phi), np.cos(phi)]])
    #     M_rot = M_rot_y@M_rot_x
    
    phi = phi / 180.0 * np.pi
    inc = inc / 180.0 * np.pi  # i =90 => edge-on
    M_rot_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                        [np.sin(phi), np.cos(phi), 0],
                        [0, 0, 1]])
    M_rot_x = np.array([[1, 0, 0],
                        [0, np.cos(inc), -np.sin(inc)],
                        [0, np.sin(inc), np.cos(inc)]])
    M_rot = M_rot_x@M_rot_z

    if pa:
        print("Rotating by the position angle.")
        #Rotate by the Position angle, measure clockwise from x'
        pa = pa / 180.0 * np.pi
        M_rot_pa = np.array([[np.cos(pa),  np.sin(pa), 0],
                            [-np.sin(pa), np.cos(pa), 0],
                            [     0,          0,      1]])
        M_rot = M_rot_pa@M_rot
    else:
        pass

    return M_rot

def get_re(sol):
    '''
    Effective radius.
    '''
    L = 2*np.pi*sol[:, 0]*sol[:, 1]**2*sol[:, 2]
    R = np.logspace(np.log10(0.5), np.log10(50), 5000)
    Lu = R.copy()
    for i in range(R.size):
        Lu[i] = np.sum(L*(1-np.exp(-(R[i])**2/(2.*sol[:, 1]**2*sol[:, 2]))))
    tLu = np.sum(L)/2.
    ii = np.where(np.abs(Lu-tLu) == np.min(np.abs(Lu-tLu)))
    return R[ii]

def density_profile(x, y, z, quantity, Nbins, rmax, 
                        spaced="logspace", rmin=None):
    """
        Calculates 3D density profile for any quantity, in spherical shells.
        Inputs:
        ----------
        x,y,z: 1N-array (dimension of length)
            (x,y,z) positions relative to the center of the cube.
        quantity: 1N-array
            1N-array with the quantity that you want to compute the sperical
            density profile.
        Nbins: int
            number of shells
        rmax: float (same dimension as positions)
            radius of the outermost shell.
        spaced: str
            Default is "logspace".
            "linear": create the radial bins linearly spaced
            "logspace": create the radial bins logarithmically spaced.
        rmin: float (same dimension as positions)
            minimum radius for the logspaced radii. Only used in this case.
        Output:
        ------------
        rho: 1D-array (dimension of quantity/length**3)
            3D radial density profile of the input quantity.
        radii: 1D-array (dimension of length)
            array with the mean radius value for each spherical
            shell where the density profile where computed.
        error: 1D-array (dimension of quantity/length**3)
            Poisson error on the density profile.
            USE WITH CARE. NEEDS TO CHECK IF IT IS FINE
        Npar: 1D-array
            Number of particles in each shell
    """

    density = []
    radii   = []
    error   = []
    Npart   = []  # Number of particles in each bin
    r = np.sqrt(x**2 + y**2 + z**2) #radial distance to the centre

    if  spaced == "logspace":
        if rmin is None:
            raise ValueError("You must provide a minimum radius for the logspaced sampling.")
        else: pass
        Bins = np.logspace(np.log10(rmin), np.log10(rmax), Nbins)
        for i in range(Bins.size-1):
            r1 = Bins[i]
            r2 = Bins[i+1]

            #particles within the shell
            shell = np.argwhere( (r > r1) & (r < r2) ).flatten()
            
            quantity_within = sum(quantity[shell])
            dVol = 4/3.0 * np.pi * (r2**3 - r1**3)
            rho = quantity_within/dVol

            radius = (r1 + r2)/2.0

            density.append(rho)
            radii.append(radius)
            error.append( np.sqrt(quantity[shell].size/dVol**2) )
            Npart.append(quantity[shell].size)

    elif  spaced == "linear":
        dr = rmax/int(Nbins)    #thickness of each shell
        
        for i in range(Nbins):
            r1 = i * dr
            r2 = r1 + dr

            #particles within the shell
            shell = np.argwhere( (r > r1) & (r < r2) ).flatten()
            
            quantity_within = sum(quantity[shell])
            dVol = 4/3.0 * np.pi * (r2**3 - r1**3)
            rho = quantity_within/dVol

            radius = (r1 + r2)/2.0

            density.append(rho)
            radii.append(radius)
            error.append( np.sqrt(quantity[shell].size/dVol**2) )
            Npart.append(quantity[shell].size)
    else:
        return ValueError("Type of spacing not valid.")
    
    return np.asarray(density), np.asarray(radii), np.asarray(error), np.asarray(Npart)

def _density_profile(positions, quantity, spaced="linear", 
                        step=None, nsample=None, rmin=None, rmax=None):
    """
        DEPRECATED UNTIL I FIND THE ERROR



        Calculates 3D density profile for any quantity, in spherical shells.
        Inputs:
        ------------
        positions: 3N-array (dimension of length)
            3N-array with positions (x,y,z) relative to the center of the cube.
        quantity: 1N-array
            1N-array with the quantity that you want to compute the sperical
            density profile.
        spaced: str
            Default is "linear".
            "linear": create the radial bins linearly spaced by step, between
            rmin and rmax.
            "logspace": create the radial bins logarithmically spaced, between
            rmin and rmax with nsamples.
        step: float (same dimension as positions)
            Default is 0.5 units of length.
            Step for which the radial bins will be evenly spaced, 
            assuming the radial bins will be linearly computed (spaced="linear").
            Note that step should not be set with nsample.
        nsample: int
            Number of points where we want to sample your profile.
            Default is 100.
        rmin: float (same dimension as positions)
            Radius of the innermost shell. 
            If None, assumes the minimum radius computed by the positions.
        rmax: float (same dimension as positions)
            Radius of the outermost shell.
            If None, assumes the maximum radius computed by the positions.
        Output:
        ------------
            rho: 1D-array (dimension of quantity/length**3)
                3D radial density profile of the input quantity.
            rmed: 1D-array (dimension of length)
                array with the mean radius value for each spherical
                shell where the density profile where computed.
            error: 1D-array (dimension of quantity/length**3)
                Poisson error on the density profile.
                USE WITH CARE. NEEDS TO CHECK IF IT IS FINE
    """
    r = np.hypot(positions[:,0], positions[:,1], positions[:,2])
    if rmin is None: 
        rmin = r.min()
    else: pass
    
    if rmax is None:
        rmax = r.max()
    else: pass
    
    
    
    if spaced=="linear" and step is None:
        step = 0.5
        bins = np.arange(rmin, rmax, step)
        
    elif spaced=="linear" and step != None:
        bins = np.arange(rmin, rmax, step)
        
    elif spaced=="logspace" and nsample is None:
        nsample = 100
        bins = np.logspace(np.log10(rmin), np.log10(rmax), nsample)
        
    elif spaced=="logspace" and nsample != None:
        bins = np.logspace(np.log10(rmin), np.log10(rmax), nsample)
        
    elif spaced=="linear" and nsample != None:
        return ValueError("spaced 'linear'  and nsample should not be set together.")
    
    elif spaced=="logspace" and step != None:
        return ValueError("spaced 'logspace'  and step should not be set together.")

    
    
    
    rho  = np.empty(bins.size-1)
    rmed = np.empty(bins.size-1)
    error = np.empty(bins.size-1)
    
    for i in range(bins.size-1):
        ii = (bins[i] < r) & (r < bins[i+1])
        quantity_within = quantity[ii]
        dVol    = (4*np.pi/3)*(bins[i+1]**3 - bins[i]**3)
        rho[i]  = quantity_within.sum()/dVol
        rmed[i] = (bins[i+1] + bins[i])/2
        error[i] = np.sqrt(quantity_within.size/dVol**2)
        
    
    return rho, rmed, error

