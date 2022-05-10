# setuptools
try:
    import setuptools
    print('Checking for setuptools... OK')
except:
    print('Checking for setuptools... NO')
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
    'pandas'
]

setuptools.setup(
    name="CompNeuroPy",
    version="0.0.1",
    description="General package for computational neuroscience with ANNarchy.",
    url="https://github.com/Olimaol/compneuropy",
    packages=setuptools.find_packages(),
    install_requires=dependencies
)
