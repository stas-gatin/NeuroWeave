# NeuroWeave
<p align="center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=2000&pause=500&center=true&vCenter=true&multiline=true&repeat=false&random=false&width=800&height=100&lines=Interweaving+Intelligence;Where+Neurons+and+Networks+Connect" alt="Typing SVG" /></a>
</p>

![Static Badge](https://img.shields.io/badge/Version-v1.0.0-green)
![Static Badge](https://img.shields.io/badge/Colaboradores-5-blue)
<img src="https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=style=flat&color=BC4E99" alt="Star Badge"/>
[![Star on GitHub](https://img.shields.io/github/stars/stas-gatin/stas-gatin.svg?style=social)](https://github.com/stas-gatin/NeuroWeave/stargazers)

> [!NOTE]
> The library is still under development!

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.



## Task list: <img src="https://media.giphy.com/media/WUlplcMpOCEmTGBtBW/giphy.gif" width="30">
- [x] Encontrar como guardar modelos de redes neuronales (libreria h5py)
- [ ] 1. Tensors
  <details>
    <summary>Click to expand more about Tensores</summary>
    Tensors a fundamental data structure used in Machine Learning for multi-dimensional matrix operations.
  </details>

- [ ] 2. Clases para las capas que conforman los modelos (grande, podría ser dividido en varias personas, o no)
- [x] 3. Guardado de modelos en un formato eficiente
- [ ] 4. Cargado y preparado de datasets ocn clases (Datasets, Dataloaders)
- [ ] 5. Visualización con Manim u otros (?)
- [ ] 6. Métodos y clases para el manejo aritmético de Tensores
- [ ] 7. Implementación de métodos con opción de ejecución el GPU (quizás, sobre consideración)

## Docs:

### Load model

Load a model from an HDF5 file.
```python
weave.loader(file_path)
```
    Parameters:
    file_path : string, the path to the file from which the model is being loaded.

    Returns:
    A dictionary with 'weights' and 'config'.

### Save model
Save a neural network model to an HDF5 file.
```python
weave.saver(model, file_path=None, overwrite=False)
```
    Parameters:
    model : model object, which must have 'weights' and 'config' attributes.
    file_path : string, the path to the file where the model will be saved.
    overwrite : bool, determines whether to overwrite the file if it already exists.

> [!WARNING]
> Set overwrite=True to overwrite it.

## Jerarquía de clases:

```mermaid
graph TD;
    weave-->neuro_storage;
    weave-->glimpse;
    weave-->optim;
    weave-->random;
    weave-->nn;
    nn-->modules;
    neuro_storage-->saver;
    neuro_storage-->loader;
```

## Collaborators

<!-- readme: collaborators -start -->
<table>
<tr>
    <td align="center">
        <a href="https://github.com/itprosta">
            <img src="https://avatars.githubusercontent.com/u/81316740?v=4" width="100;" alt="itprosta"/>
            <br />
            <sub><b>ITPROSTA</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Shillianne">
            <img src="https://avatars.githubusercontent.com/u/148450883?v=4" width="100;" alt="Shillianne"/>
            <br />
            <sub><b>Shillianne</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/hugourmaz">
            <img src="https://avatars.githubusercontent.com/u/149888695?v=4" width="100;" alt="hugourmaz"/>
            <br />
            <sub><b>Hugourmaz</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/CARLOSMOLERA">
            <img src="https://avatars.githubusercontent.com/u/152264006?v=4" width="100;" alt="CARLOSMOLERA"/>
            <br />
            <sub><b>CARLOSMOLERA</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/paatriiperezz">
            <img src="https://avatars.githubusercontent.com/u/152264650?v=4" width="100;" alt="paatriiperezz"/>
            <br />
            <sub><b>Patricia Pérez Ferre</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/stas-gatin">
            <img src="https://avatars.githubusercontent.com/u/155986458?v=4" width="100;" alt="stas-gatin"/>
            <br />
            <sub><b>Stanislav Gatin</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: collaborators -end -->


