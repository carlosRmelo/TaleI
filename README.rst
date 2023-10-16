TaleI
========
This repo is intended to present and make the first-part results of my Ph.D. publicly available.
The subject is how to combine, self-consistently, stellar dynamics and galaxy-galaxy strong gravitational lensing.
The associated paper is on the way. Until then, check out our previous work regarding probes of General Relativity using combined modelling of stellar dynamics and strong gravitational lensing,  [Melo-Carneiro et al. (2023)](<https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.1613M/abstract>).




Folders
-------
Each folder represents one of the mock galaxies based on the TNG50 subhalos. The results represent the final model achieved by dynamical-only modelling, lens-only modelling or joint modelling. 

- ``data``: The mock (lensing and kinematics) data.
- ``model``: The results related to a specific model used to fit the data.
	* normal: means that a small FoV was used to generate the lensing data;
	* bigger: means that a bigger FoV was used to generate the lensing data;
	* shear: means that an external shear was added to the lensing model;
	* no_shear: means that no external shear was added to the lensing model;
- ``phase``: Result of a specific phase of the Pipeline. Final results are based on phase5.
- ``JAM``: Dynamical-only model results.
- ``Lens``: Lens-only modelling.
- ``dyLens``: Combined (lens+dynamics) modeling.
- ``Analysis``: Final analysis containing the measurements of interest within a given radius. 
	* Fiducial: Analysis was performed using the fiducial method, i.e., the median of the one-dimensional posterior distribution of the parameters. 
 

Files
-----
