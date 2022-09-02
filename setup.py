# setuptools
try:
    import setuptools
except:
    print('Error : Python package "setuptools" is required.')
    assert False

dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    'cython',
    'sympy',
    'hyperopt',
    'ANNarchy',
    'pandas',
    'sbi',
    'torch'
]

setuptools.setup(
    name="CompNeuroPy",
    version="0.0.5",
    license="MIT",
    author="Oliver Maith",
    author_email="oli_maith@gmx.de",
    description="General package for computational neuroscience with ANNarchy.",
    url="https://github.com/Olimaol/compneuropy",
    packages=setuptools.find_packages(),
    install_requires=dependencies
)
