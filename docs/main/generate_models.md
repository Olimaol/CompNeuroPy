## Introduction
One can create a CompNeuroPy-model using the `CompNeuroModel` class. The `CompNeuroModel` class takes as one argument the `model_creation_function`. In this function a classical ANNarchy model is created (populations, projections). The `CompNeuroModel` class only adds a framework to the model. Neccessary for a CompNeuroPy-model is to define unique names for all populations and projections. Models are created in three steps:

1. **model initialization**: the initialization of the `CompNeuroModel` object, initializes the framework of the model without creating the ANNarchy objects (populations, projections)
2. **model creation**: create the ANNarchy objects (populations, projections), i.e., run the `model_creation function`
3. **model compilation**: compile all created models

## Example
<pre><code>from CompNeuroPy import CompNeuroModel
my_model = CompNeuroModel(model_creation_function=create_model,  ### the most important part, this function creates the model (populations, projections)
                          model_kwargs={'a':1, 'b':2},           ### define the two arguments a and b of function create_model
                          name='my_model',                       ### you can give the model a name
                          description='my simple example model', ### you can give the model a description
                          do_create=True,                        ### create the model directly
                          do_compile=True,                       ### let the model (and all models created before) compile directly
                          compile_folder_name='my_model')        ### name of the saved compilation folder
</code></pre>

The following function could be the corresponding model_creation_function:
<pre><code>from ANNarchy import Population, Izhikevich
def create_model(a, b):
    pop = Population(geometry=a, neuron=Izhikevich, name='Izh_pop_a') ### first population, size a
    pop.b = 0                                                         ### some parameter adjustment
    Population(geometry=b, neuron=Izhikevich, name='Izh_pop_b')       ### second population, size b
</code></pre>
Here, two populations are created (both use built-in Izhikevich neuron model of ANNarchy). The function does not require a return value. It is important that all populations and projections have unique names.

A more detailed example is available in the [Examples](../examples/generate_models.md).

::: CompNeuroPy.generate_model.CompNeuroModel