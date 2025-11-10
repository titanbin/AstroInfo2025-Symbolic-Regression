# Globular cluster analytical radial profile solutions using $\Phi$-SO

This repository contains the work done during the hackaton week of the AstroInfo 2025 school.

This project was centered around the use of the $\Phi$-SO [python package](https://physo.readthedocs.io/) to infer analytical formulae for radial profiles of different globular cluster properties.
The two properties that were used was the azimuthal velocity profile $v_\phi(r)$ and azimuthal velocity dispersion profile ($\sigma_{v_\phi}(r)$ ).

The repository contains a general script `SR_profile.py` that runs $\Phi$-SO on these profiles, as well as two jupyter notebooks to analyse the results.
- `Cluster property SR.ipynb` analyses the result of general analtyical solutions to radial profiles given globular clusters with different physical properties as input.
- `Time evolution SR.ipynb` analyses the result of general analtyical solutions to radial profiles given the same globular cluster at different timestamps as input.
