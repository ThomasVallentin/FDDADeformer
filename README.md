# Fast and Deep Deformation Approximation (FDDA) for Autodesk Maya

An implementation of Fast and Deep Deformation Approximation (Bailey 2018) for Autodesk Maya. This project has been developped as the end project for the Deep Learning course at Université Gustave Eiffel (France).

https://user-images.githubusercontent.com/43669549/220860659-d9e0d540-26ef-41ea-8dfd-ba3868810fb2.mov

**Legend:** *Green=equal, Blue=+-1e-4 cm, Red=+-0.5 cm*

## Dependencies 

This project has been written in python 3.9 and rely on the following libraries :

- numpy
- keras
- jupyter -> Only if you want to train the networks using a Jupyter notebook

which can be installed to Maya using the following command :

```
/path/to/your/maya/install/bin/mayapy -m pip install tensorflow jupyter 
```

## How to use 

The main feature of the project is the FDDADeformer, a deformer that learns and applies the offsets between a hard weighted skinCluster and a complete deformation rig. Each deformer uses several deep learning networks (one for each joint bound to the skinCluster) which needs to be trained on a set of poses.

### Recording the dataset

WIP section
```python
from fdda import recording
 
recorder = recording.Recorder("meshToLearn")
recorder.initialize()
recorder.record(samples=200)
recorder.finalize()

# The mesh name is used as the name of the dataset if no name is passed
recorder.save("/path/to/your/dataset/directory", name="nameOfTheNetwork")
```

### Training the models

*WIP section*

### Using the deformer

*WIP section*

### Extra content

The project contains a second plugin "distanceToColor" that takes a mesh and a refMesh as input and sets the vertexColors of the outputMesh based on the distance between the two meshes. This can be useful to detect differences between the approximation and the ground truth deformations.


## Roadmap

This project is still work in progress, here are the implemented features yet and the ones that still need to be implemented :

- [x] Segment the mesh into subsets
- [x] Record gaussian sampled poses
- [x] Train the models
- [x] Load the models in the deformer and use them to apply the deformation
- [ ] Data simplification : Localize the deformation to the main joints of each subset
- [x] Input reduction : Each model only receive the relevant samples based on the joints that have some influence on the model's subset
- [ ] Output reduction : Use PCA to compute correlations between vertices and reduce the output count
- [ ] Model Count Reduction : Remove small subsets using a threshold
