# ITAM
Three dimensional implementation of the Iterative Translation Approximation Method (ITAM).

ITAM is an algorithm developed by Michael Shields and collaborators (https://doi.org/10.1016/j.probengmech.2011.04.003) to make fast realizations of a one-dimensional non-Gaussian stationary stochastic process. We extended it to the three dimensional case, and applied it to three-dimensional cosmological fields. See the corresponding paper ....... for more details.

The code was tested on python 3.6.9. Its main dependencies are numpy and scipy. For the generation of the covariance matrix from many realizations we suggest to install pathos (https://pypi.org/project/pathos/), otherwise only a single core is used.

The lookup.py module requires also CLASS (https://lesgourg.github.io/class_public/class.html), but it is not compulsory as long as one provides lookup tables for the inverse CDF of the target field and the target power spectrum. If the one-point mapping you want to use is analytical, you can easily modify the logITAM.py module, which implements the lognormal field case.

See the quickstart.ipynb notebook for an example and further details.

If you find this code useful, please cite .........
