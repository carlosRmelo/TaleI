TaleI
========
This repo is intended to present and make the first-part results of my Ph.D. publicly available.
The subject is how to combine, self-consistently, stellar dynamics and galaxy-galaxy strong gravitational lensing.
The associated paper is on the way. Until then, check out our previous work regarding probes of General Relativity using combined modelling of stellar dynamics and strong gravitational lensing,  `Melo-Carneiro et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.520.1613M/abstract>`_. 

-------------------------------------------------------------------------------

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
- ``codes``: Codes used to run the pipelines, as well as codes to control the job submission on the LNCC supercomputer.
- ``config``: Contains configuration files which customize default **PyAutoLens**.
 

Files
-----

- ``Analysis 2D - Distribution - bigger FoV.ipynb``: Analysis of the projected quantities using the distributed method and a larger FoV.
- ``Analysis 2D - Distribution - normal FoV.ipynb``: Analysis of the projected quantities using the distributed method and a normal FoV.
- ``Analysis 2D - Fiducial - bigger FoV.ipynb``: Analysis of the projected quantities using the fiducial method and a larger FoV.
- ``Analysis 2D - Fiducial - normal FoV.ipynb``: Analysis of the projected quantities using the fiducial method and a normal FoV.

- ``Analysis 3D - Distribution - bigger FoV.ipynb``: Analysis of the intrinsic quantities using the distributed method and a larger FoV.
- ``Analysis 3D - Distribution - normal FoV.ipynb``: Analysis of the intrinsic quantities using the distributed method and a normal FoV.
- ``Analysis 3D - Fiducial - bigger FoV.ipynb``: Analysis of the intrinsic quantities using the fiducial method and a larger FoV.
- ``Analysis 3D - Fiducial - normal FoV.ipynb``: Analysis of the intrinsic quantities using the fiducial method and a normal FoV.

- ``Analysis - Slope - Distribution - bigger Fov.ipynb``: Analysis of the total density slope using the distributed method and a larger FoV.
- ``Analysis - Slope - Distribution - normal Fov.ipynb``: Analysis of the total density slope using the distributed method and a normal FoV.
- ``Analysis - Slope - Fiducial - bigger Fov.ipynb``: Analysis of the total density slope using the Fiducial method and a larger FoV.
- ``Analysis - Slope - Fiducial - normal Fov.ipynb``: Analysis of the total density slope using the Fiducial method and a normal FoV.

- ``Analysis_Distribution.py``: Used to compute the results using the distributed method.
- ``Analysis_Fiducial.py``: Used to compute the results using the fiducial method.

- ``Fiducial_density_slope.py``: Used to compute the total density slope using the density profile derived from the fiducial method.
- ``Distribution_density_slope.py``: Used to compute the total density slope using the density profile derived from the distributed method.

- ``results.rst``: Results of the non-linear sample. Median and 68% credible interval.
- ``description.json``: Description of the quantities present in quantities.json.
- ``quantities.json``: Results after the analysis (integrated quantites).

Contact
-------

If you find anything interesting or have any questions/suggestions, please keep in touch:
`carlos.melo@ufrgs.br <mailto:carlos.melo@ufrgs.br>`_




