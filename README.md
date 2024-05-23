# NeuroWeave
<p align="center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=2000&pause=500&center=true&vCenter=true&multiline=true&repeat=false&random=false&width=800&height=100&lines=Interweaving+Intelligence;Where+Neurons+and+Networks+Connect" alt="Typing SVG" /></a>
</p>

![Static Badge](https://img.shields.io/badge/Version-v1.0.0-green)
![Static Badge](https://img.shields.io/badge/Colaboradores-5-blue)
<img src="https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=style=flat&color=BC4E99" alt="Star Badge"/>
![GitHub Repo stars](https://img.shields.io/github/stars/stas-gatin/NeuroWeave)

NeuroWeave is a powerful and intuitive library for creating and training neural networks. With NeuroWeave, you can quickly build complex models, experiment with architectures, and adjust hyperparameters while maintaining control over the training process details.

## Instalation
To install the current release:
```
$ pip install neuroweave
```

> [!IMPORTANT]
> If you found a bug, please contact the repository creators.

# FAQS

## How to save and load model?

### Load model

Load a model from an HDF5 file.
```python
model = weave.loader(file_path)
```
    Parameters:
    file_path : string, the path to the file from which the model is being loaded.

    Returns:
    A dictionary with 'weights' and 'config'.

### Save model
Save a neural network model to an HDF5 file.
```python
weave.saver(model, file_path='path/to/save', overwrite=False)
```
    Parameters:
    model : model object, which must have 'weights' and 'config' attributes.
    file_path : string, the path to the file where the model will be saved.
    overwrite : bool, determines whether to overwrite the file if it already exists.

> [!WARNING]
> Set overwrite=True to overwrite the model file.

## Task distribution
Implementación de las clases de capas que permitirán manipular los tensores dentro de los módulos de IA, permitiendo la funcionalidad esencial para que cualquier usuario pueda construir sus módulos personalizados. La tarea será realizada por Gabriel Niculescu Ruso y Carlos Molera Canals. Esto serían clases que conformarían lo esencial para transformar tensores según diferentes normas dentro de los distintos modelos de IA. Dada la gran diversidad de capas posibles y todas sus dependencias, esta tarea presenta gran cantidad de contenido para que ambos integrantes tengan múltiples clases que realizar.  
 
Adición de los métodos de optimización a las clases mencionadas para que puedan trabajar en la GPU, dependiendo de los requerimientos del proyecto. De ser realizada correctamente, la velocidad de los cálculos se verá acelerada superlativamente, incrementando la utilidad de la librería. Esta tarea será realizada por Gabriel Niculescu Ruso. 
 
El guardado de modelos en ficheros de la forma más eficiente posible, sean binarios u otras extensiones más efectivas para los requerimientos de este proyecto. En esto se incluye el tratado de archivos que actúen como dataset para cargarlos y manejarlos de forma eficiente de manera que dichos archivos puedan ser aprovechados por los distintos módulos. Esta tarea será realizada por Stanislav Gatin . 
 
Visualización del proceso de aprendizaje de los módulos de IA hechos por los usuarios mediante la librería Manim u otras que se ajusten a las necesidades de este proyecto. Las visualizaciones permitirán obtener una idea intuitiva de que realiza la red neuronal y de cómo se obtienen los resultados finales que se producen. Esta tarea será realizada por Patricia Pérez Ferre. 
 
Desarrollo de los métodos aritméticos para el manejo de tensores junto. Esta tarea será esencial para poder utilizar las propiedades matemáticas que tienen los tensores en el código que generemos, así como de dar funcionalidad esencial a los usuarios para que construyan sus propias clases con funcionalidad completa. Esta tarea será completada por Hugo Urbán Martínez en conjunto con Stanislav Gatin. 
 
Implementación de la clase del Tensor y todos sus métodos. Esta clase es esencial para la realización adecuada del resto de clases que trabajan con tensores, por lo que esta tarea la realizarán todos los integrantes del equipo para adquirir conocimiento de primera mano de su funcionamiento.