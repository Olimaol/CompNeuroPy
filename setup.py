# setuptools
try:
    import setuptools
except:
    print('Error : Python package "setuptools" is required.')
    assert False

dependencies = [
    "torch",
    "sbi",
    "pandas",
    "hyperopt",
    "sympy",
    "cython",
    "matplotlib",
    "scipy",    
    "numpy",
]

setuptools.setup(
    name="CompNeuroPy",
    version="0.0.13",
    license="MIT",
    author="Oliver Maith",
    author_email="oli_maith@gmx.de",
    description="General package for computational neuroscience with ANNarchy.",
    url="https://github.com/Olimaol/compneuropy",
    packages=setuptools.find_packages(),
    install_requires=dependencies,
    include_package_data=True,
    package_data={'': ['*.csv']},
)
