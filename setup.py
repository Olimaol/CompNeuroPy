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
    'pandas'
]

setuptools.setup(
    name="CompNeuroPy",
    version="0.0.1",
    license="MIT",
    author="Oliver Maith",
    author_email="oli_maith@gmx.de",
    packages=setuptools.find_packages('CompNeuroPy'),
    package_dir={'': 'CompNeuroPy'},
    description="General package for computational neuroscience with ANNarchy.",
    url="https://github.com/Olimaol/compneuropy",
    install_requires=dependencies
)
