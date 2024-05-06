from setuptools import setup, find_packages

packages = {"" : "src"}
for package in find_packages("src"):
    packages[package] = "src"

setup(packages = packages.keys(),
      package_dir = {"" : "src"},
      name = 'gppm',
      version = '2.0.0',
      author = 'Marco Lorenzi',
      author_email = 'marco.lorenzi@inria.fr',
      description = 'Gaussian process-based disease progression modelling and time-shift estimation',
      long_description = 'This software is based on the publication: Probabilistic disease progression modeling to characterize diagnostic uncertainty: Application to staging and prediction in Alzheimers disease. Marco Lorenzi, Maurizio Filippone, Daniel C. Alexander, Sebastien Ourselin. Neuroimage. 2019 Apr 15,190:56-68. doi: 10.1016/j.neuroimage.2017.08.059. Epub 2017 Oct 24. HAL Id : hal-01617750',
      license = 'proprietary Inria license',
    )
