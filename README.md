# Introduction

Since the introduction of AlexNet in 2012, deep convolutional neural networks (CNN) have become the dominating approach for image classification. Various new architectures have been proposed since then, including VGG, NiN, Inception, ResNet, DenseNet, and NASNet. At the same time, we have seen a steady trend of model accuracy improvement. For example, the top-1 validation accuracy on ImageNet has been raised from 62.5% (AlexNet) to 82.7% (NASNet-A).

Pattern recognition and image classification has surprisingly become one of the efficient methods of malware detection/classification. Back in 2011, the researchers from University of California (Santa Barbara, California, USA) - L. Nataraj, S. Karthikeyan, G. Jacob, and B. S. Manjunath - proposed a simple yet effective method for visualizing and classifying malware using image processing techniques. Malware binaries can be visualized as gray-scale images, with the observation that for many malware families, the images belonging to the same family appear very similar in layout and texture. Motivated by this visual similarity, a classification method using standard image features had been proposed by the researcher team. Neither disassembly nor code execution is required for such a classification. Preliminary experimental results were quite promising with 98% classification accuracy on a malware database of 9,458 samples with 25 different malware families (see their public paper per https://vision.ece.ucsb.edu/sites/default/files/publications/nataraj_vizsec_2011_paper.pdf for more info). 

# DL With Attention

TBD

# Dataset Description

The classical MalImg dataset (https://paperswithcode.com/paper/malware-images-visualization-and-automatic) prepared by the inventors of Malware Visualization and Automatic Classification method (see their paper describing such a method of malware detection/classification via greyscale image visualizing and classification as of 2011, per https://vision.ece.ucsb.edu/sites/default/files/publications/nataraj_vizsec_2011_paper.pdf). 

The dataset replica used in our experiments has been downloaded from one of the unofficial mirrors of such a dataset published on Kaggle, per https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset9010/ 

The challenge manifested by this dataset is to tackle the multi-class classification problem to predict the malware application class represented by a particular gray-scale bitmap image in the dataset.


# How to Install the Dataset

You should download the dataset from https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset9010

It will appear as *datast9010.zip* on your computer. Just unzip it to any local folder of your choice.

Once you unzip it, you will see the following folder structure in it

```*<your_local_folder>/dataset_9010/malimg_dataset/{some_child_subfolders}*```

You will have to move malimg_dataset/{some_child_subfolders} to the subfolder where you put the source code of Jupyter notebooks.


# References

The replica of MalImg dataset downloaded from https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset9010

The blog posts and articles referred / reused in the experiments are listed below

- https://vision.ece.ucsb.edu/sites/default/files/publications/nataraj_vizsec_2011_paper.pdf
