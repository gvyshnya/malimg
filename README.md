# Preface

This repo summarizes the results of the joint effort of the researcher group (George Vyshnya, Denys Frolov and Co). The main purpose of such an effort was to demonstrate that the novel DL network architectures with attention can improve the results of the malware detection by now-classical Malware Visualization and Automatic Classification method.

# Introduction

Since the introduction of AlexNet in 2012, deep convolutional neural networks (CNN) have become the dominating approach for image classification. Various new architectures have been proposed since then, including VGG, NiN, Inception, ResNet, DenseNet, and NASNet. At the same time, we have seen a steady trend of model accuracy improvement. For example, the top-1 validation accuracy on ImageNet has been raised from 62.5% (AlexNet) to 82.7% (NASNet-A).

Pattern recognition and image classification has surprisingly become one of the efficient methods of malware detection/classification. Back in 2011, the researchers from University of California (Santa Barbara, California, USA) - L. Nataraj, S. Karthikeyan, G. Jacob, and B. S. Manjunath - proposed a simple yet effective method for visualizing and classifying malware using image processing techniques. Malware binaries can be visualized as gray-scale images, with the observation that for many malware families, the images belonging to the same family appear very similar in layout and texture. Motivated by this visual similarity, a classification method using standard image features had been proposed by the researcher team. Neither disassembly nor code execution is required for such a classification. Preliminary experimental results were quite promising with 98% classification accuracy on a malware database of 9,458 samples with 25 different malware families (see their public paper per https://vision.ece.ucsb.edu/sites/default/files/publications/nataraj_vizsec_2011_paper.pdf for more info). 

# DL With Attention

TBD

# Dataset Description

The classical MalImg dataset (https://paperswithcode.com/paper/malware-images-visualization-and-automatic) prepared by the inventors of Malware Visualization and Automatic Classification method (see their paper describing such a method of malware detection/classification via greyscale image visualizing and classification as of 2011, per https://vision.ece.ucsb.edu/sites/default/files/publications/nataraj_vizsec_2011_paper.pdf). 

The dataset replica used in our experiments has been downloaded from one of the unofficial mirrors of such a dataset published on Kaggle, per https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset9010/ 

The challenge manifested by this dataset is to tackle the multi-class classification problem to predict the malware application class represented by a particular gray-scale bitmap image in the dataset.

# DL Experiment Setup 

In the scope of this case study, we aimed at comparing the performance of classical CNN DL model with the performance of the more novel attention-capable neural network architectures. As a part of the experiment, the following models have been trained and evaluated

- A classical CNN model
- Swin Transformer Models (V1 model and two different models of V2 generation)
- ResNet model (namely, its ResNetD variation)
- CoAtNet model (CoAtNet is a hybrid neural network architecture to combine the best of CNN and Attention-based transformer models)

Every model was trained and evaluated within the framework outlined below

- Cross-entropy used as a loss function for the models at the training time
- Accuracy on the validation set used as a performance metric
- The technique of the dynamic decrease of the learning rate of a DL model on the plateau used to get the additional performance edge
- Early stopping of the model training upon the reach to the performance gain threshold used both to address the overfitting pitfall and optimize the model training time

# Classic CCN Model

We have invented a custom CCN model tailored for the specific multi-class classification problem for the MalImg dataset used in this case study.

# Swin Transformer V1 Model

We have reused the model implementation from https://keras.io/examples/vision/swin_transformers/. Such an implementation is based on the public paper of “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows” (https://arxiv.org/pdf/2103.14030.pdf). 

# Swin Transformer V2 Models

Keras SwinTransformerV2 from Keras_cv_attention_models Python package contains the implementation of the model proposed in “Swin Transformer V2: Scaling Up Capacity and Resolution” (https://arxiv.org/pdf/2111.09883.pdf). 
Swin Transformer V2 architecture designates the major improvement in the large-scale models in computer vision vs. the original Swin Transforme V1 architecture. The time distance between Swin Transformer V1 and V2 is about one year only yet the headway has been tremendous. As a part of the current experiment, we would like to see how the DL model architecture invented for large-scale image pattern recognition would work on images of relatively small size.

# CoAtNet Model

CoAtNet is a hybrid neural network architecture where the best of CNN and Attention-based transformer models is conjugated in a single model architecture. It has been invented Zihang Dai, Hanxiao Liu, Quoc V. Le, and Mingxing Tan from Google Research team (their inventions published in https://arxiv.org/pdf/2106.04803.pdf).

Transformers have attracted increasing interests in computer vision/image pattern recognition. However, they still fall behind state-of-the-art convolutional networks. The authors of CoAtNet demonstrated that while Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. To effectively combine the strengths from both architectures, they invented CoAtNets, a family of hybrid models built from two key insights: 
(1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention. 
(2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency.

In the scope of this case study, the implementation of CoatNet from Keras_cv_attention_models Python package (per https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/convnext) has been used.

# ResNetD Model

Keras ResNetD includes ResNet implementation with a number of data preprocessing and model training/architecture improvements proposed and evaluated in “Bag of Tricks for Image Classification with Convolutional Neural Networks” (https://arxiv.org/pdf/1812.01187.pdf) 

# Modelling results

The table below summarizes the results of the model training and evaluation experiment

TBD

As we can see, the hybrid CoAtNet DL architecture demonstrated the best performance (on the validation set) at tackling the malware classification problem as per this case study. The relatively small size of the images in the dataset did not let Swin Transformer-style models to add edge as they are tailored to the ‘attentive’ pattern recognition at the large image processing scenarios.

# How to Install the Dataset

You should download the dataset from https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset9010

It will appear as *datast9010.zip* on your computer. Just unzip it to any local folder of your choice.

Once you unzip it, you will see the following folder structure in it

```<your_local_folder>/dataset_9010/malimg_dataset/{some_child_subfolders}```

You will have to move malimg_dataset/{some_child_subfolders} to the subfolder where you put the source code of Jupyter notebooks.


# References

The replica of MalImg dataset downloaded from https://www.kaggle.com/datasets/keerthicheepurupalli/malimg-dataset9010

The blog posts and articles referred / reused in the experiments are listed below

- https://vision.ece.ucsb.edu/sites/default/files/publications/nataraj_vizsec_2011_paper.pdf
- https://keras.io/examples/vision/swin_transformers/
