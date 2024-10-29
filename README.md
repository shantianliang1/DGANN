# Few-Shot Image Classification via Double-Ended Graph Augmentation Neural Networks
## Abastract
Few-Shot Learning (FSL) aims to effectively predict unlabelled data using only a limited number of labelled samples. This paper addresses the challenges of existing FSL methods, particularly meta-learning-based approaches prone to overfitting and Graph Neural Network (GNN)-based methods that struggle to fully utilize image data. To overcome these limitations, we propose a novel GNN model, the Double-Ended Graph Augmentation Neural Network (DGANN). The DGANN model introducesa double-ended augmentation module to enhance the diversity and expressiveness of image features. This is achieved through front-end image generation and back-end image augmentation techniques. Furthermore, a hierarchical inference module is designed to infer node features and obtain enhanced data sample features, improving image classification accuracy. Experimental results on benchmark
datasets, miniImageNet and CIFAR-FS, demonstrate that DGANN significantly outperforms traditional FSL methods and GNN-based approaches, achieving up to 14% and 63% improvements in classification accuracy for 5-way 1-shot and 10-way 5-shot tasks, respectively. The proposed DGANN model exhibits higher accuracy and better generalization ability, especially when dealing with small-size image datasets, underscoring its potential for applications in image classification and related fields.

## Requirements
CUDA Version：11.6

Python：3.8.19

    pip install -r requirements.txt


## key algorithms
Our algorithm first uses image data to process the dataset, which can be found in the `DataAugmentation` folder. Then, it performs image classification tasks and simultaneously enhances data features, which can be found in the `DataAugmentation`, `DatasetChoose` folders, and `train.py` . Our model is located in the Model folder and includes the backbone and GNN-model.



1. please download the dataset miniImageNet/cifar-fs like the form of （train/val/test// CSV.）
2. please begin the first data generation
3. train
