From PyPI using pip:

```
pip install CompNeuroPy
```

With downloaded source code; using pip in the top-level directory of the downloaded source code:

```
pip install .
```

or in development mode:

```
pip install -e .
```

You must install ANNarchy separately, best **after** CompNeuroPy.

```
pip install ANNarchy
```

Optional install torch, sbi, and hyperopt to be able to use [OptNeuron](./main/optimize_neuron.md)
```
pip install torch sbi hyperopt
```