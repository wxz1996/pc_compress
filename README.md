### Introduction

This repo of the research paper, **Lossy Geometry Compression of 3D Point Cloud via an Adaptive Octree-guided Network.** （ICME 2020 Oral)

### Abstract :



## Guide:

1.Prerequisites:

python3.6+ Tensorflow1.13 with CUDA10.0

2.Downloading the dataset:

We use the ShapeNet in our experiments, which are available below:

[Shapenet dataset](https://www.shapenet.org/)

3.Build 
the compile command can be find in ```bash.sh```, you should adjust it slightly to compile each operation

4.Training
```
cd src
python train.py
```
4.Compression/Decompression
```
python compress.py
python decompress.py
```

