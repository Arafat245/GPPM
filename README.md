# GP progression model

## The problem
Longitudinal dataset of measurements from neurodegenerative studies generally lack of a well-defined temporal reference, since the onset of the pathology may vary across individuals according to genetic, demographic and environmental factors. Therefore, age or visit date information are biased time references for the individual longitudinal measurements. There is a critical need to define the AD evolution in a data-driven manner with respect to an absolute time scale associated to the natural history of the pathology.

## The solution: GPPM ;)

The [Gaussian Process Progression Model (GPPM)](https://gitlab.inria.fr/epione/GP_progression_model_V2) is based on the probabilistic estimation of biomarkers’ trajectories and on the quantification of the uncertainty of the predicted individual pathological stage. The inference framework accounts for a time reparameterization function, encoding individual differences in disease timing and speed relative to the fixed effect. 

Thanks to the probabilistic nature of GPPM, the resulting long-term disease progression model can be used as a statistical reference representing the transition from normal to pathological stages, thus allowing probabilistic diagnosis in the clinical scenario. The model can be further used to quantify the diagnostic uncertainty of individual disease severity, with respect to missing measurements, biomarkers, and follow-up information.

GPPM has three key components underlying its methodology: 1) it defines a non-parametric, Gaussian process, Bayesian regression model for individual trajectories, 2) introduces a monotonicity information to impose some regular behaviour on the trajectories, and 3) models individual time transformations encoding the information on the latent pathological stage.

## A variety of applications

The model was originally published in [NeuroImage 2017](https://pubmed.ncbi.nlm.nih.gov/29079521/), and demonstrated on a large cohort of amyloid positive Alzheimer's disease individuals.

GPPM has been extended in recent years, and is now capable of disentangling spatio-temporal disease trajectories from collections of [high-dimensional brain images](https://doi.org/10.1016/j.neuroimage.2019.116266), and imposing a variety of [biology-inspired constraints on the biomarker trajectories](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_5).

### Some scientific literature on the GPPM methods
   - The theory of (deep) GP regression under constraints on the derivatives:

      *[M. Lorenzi and M. Filipponr. In ICML 2018: International Conference on Machine Learning, PMLR 80:3227-3236, 2018](http://proceedings.mlr.press/v80/lorenzi18a.html)*

  - Modeling biomarkers' trajectories in Alzheimer's disease: 

      *[M. Lorenzi, M. Filippone G.B. Frisoni, D.C. Alexander, S. Ourselin. Probabilistic disease progression modeling to characterize diagnostic uncertainty: Application to staging and prediction in Alzheimer's disease, NeuroImage 2017](https://pubmed.ncbi.nlm.nih.gov/29079521/)*

  - Modeling the dynamics of amyloid propagation across brain networks: 

      *[S. Garbarino and M. Lorenzi, In IPMI 2019: Information Processing in Medical Imaging pp 57-69](https://arxiv.org/abs/1901.10545)*, 

      *[S. Garbarino and M. Lorenzi, NeuroImage 2021](https://pubmed.ncbi.nlm.nih.gov/33823273/)*

  - Spatio-temporal analysis of multimodal changes from time series of brain images:

      *[C. Abi Nader, N. Ayache, P. Robert and M. Lorenzi. NeuroImage 2020](https://pubmed.ncbi.nlm.nih.gov/31648001/)*

#
 The software

GPPM and GPPM-DS enable the following analyses: 

- [GPPM] reoconstruct the profile of biomarkers evolution over time, 
- [GPPM] quantify the subject-specific disease severity associated with the measurements each individual (missing observations are allowed),
- [GPPM] estimate the ordering of the biomarkers from normal to pathological stages,
- [GPPM-DS] specify prior hypothesis about the causal interaction between biomarkers,
- [GPPM-DS] data-driven estimation of the interaction parameters, 
- [GPPM-DS] data-driven comparison between different hypothesis to identify the most plausible interaction dynamics between biomarkers,
- [GPPM-DS] model personalisation to simulate and predict subject-specific biomarker trajectories,

The software comes with a simple installation (see section Installation) and with an easy interface. 

An example of the basic usage of GPPM on synthetic and real data is available here:

[Jupyter notebook](https://gitlab.inria.fr/epione/GP_progression_model_V2/-/blob/develop/notebooks/GPPM_walkthrough.ipynb)

[Colab notebook](https://colab.research.google.com/drive/1JcouPj4KzOC_klOa2uwRvNHVtdjEensz?usp=sharing)

The software is freely distributed for academic purposes. All the commercial rights are owned by Inria.

# Contacts

If you need of any support or you want to contribute to GP progression modeling, feel free to contact marco.lorenzi-at-inria.fr

# Installation

The `setup.py` file  uses setuptools to package the python code, according to the `conda/environment.yaml` file.
To create the conda environment of gppm, type the following command line

 ```console

  conda env create -f conda/environment.yaml
  conda activate gppm

 ```

And you are ready to use gppm!

# Creating the Python package
An additional option is to  create a Python package that can be used in the gppm environment.
Three modes are possible:

1. A develop mode.
   Using this mode, all local modifications of source code will be considered in your Python interpreter (when restarted) without any post-installation.
   This is particularly useful when adding new features.
   To install this package in develop mode, type the following command line

   ```console

   python setup.py develop --prefix=${CONDA_PREFIX}

   ```

   Once this step is done, type the following command line for running tests

   ```console

   nosetests test -x -s -v

   ```

   Note that this require to have installed `nose` in your environment.

2. An install mode.
   Using this mode, all local modifications of source code will **NOT** be considered in your Python interpreter (when restarted).
   To consider your local modifications of source code, you must install the package once again.
   This is particularly useful when you want to install a stable version in one Conda environment while creating another environment for using the develop mode.
   To install this package in develop mode, type the following command line

   ```console

   python setup.py install --prefix=${CONDA_PREFIX}

   ```

3. A release mode.
   Using the mode, a Conda package will be created from the install mode and can be distributed with Conda.
   To build this package with Conda (with `conda-build` installed), type the following command line

   ```console

   conda build conda/recipe -c pytorch -c defaults --override-channels

   ```

   Then, you can upload the generated package or just install it.
   To install this conda package, type the following command line
   
   ```console

   conda install gppm -c local -c pytorch -c defaults --override-channels

   ```


  With the previous modes, the Conda environment doesn't know that this python package has been installed.
  But with this method, the `gppm` will appear in your Conda package listing.
