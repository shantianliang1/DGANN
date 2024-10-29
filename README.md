# Few-Shot Image Classification via Double-Ended Graph Augmentation Neural Networks
## Abastract
Few-Shot Learning (FSL) aims to effectively predict unlabelled data using only a limited number of labelled samples. This paper addresses the challenges of existing FSL methods, particularly meta-learning-based approaches prone to overfitting and Graph Neural Network (GNN)-based methods that struggle to fully utilize image data. To overcome these limitations, we propose a novel GNN model, the Double-Ended Graph Augmentation Neural Network (DGANN). The DGANN model introducesa double-ended augmentation module to enhance the diversity and expressiveness of image features. This is achieved through front-end image generation and back-end image augmentation techniques. Furthermore, a hierarchical inference module is designed to infer node features and obtain enhanced data sample features, improving image classification accuracy. Experimental results on benchmark
datasets, miniImageNet and CIFAR-FS, demonstrate that DGANN significantly outperforms traditional FSL methods and GNN-based approaches, achieving up to 14% and 63% improvements in classification accuracy for 5-way 1-shot and 10-way 5-shot tasks, respectively. The proposed DGANN model exhibits higher accuracy and better generalization ability, especially when dealing with small-size image datasets, underscoring its potential for applications in image classification and related fields.

## Requirements
CUDA Version：11.6
Python：3.8.19
```
pip install -r requirements.txt
```


## Key algorithms
Our algorithm first uses image data to process the dataset, which can be found in the `DataAugmentation` folder. Then, it performs image classification tasks and simultaneously enhances data features, which can be found in the `DataAugmentation`, `DatasetChoose` folders, and `train.py` . Our model is located in the Model folder and includes the backbone and GNN-model.


## Dataset
### Download the dataset
```
├── mini-imagenet
    ├── test.csv
    ├── train.csv
    ├── val.csv
    ├── images
        ├── n0153292900000006.jpg
        ├── ...
        ├── n1313361300001299.jpg
├── cifar-fs
    ├── data
        ├── apple
            ├── apple_s_000022.png
            ├── ...
        ├── ...
        ├── worm
            ├── ascaris_lumbricoides_s_000016.png
            ├── ...
    ├── splits
        ├── bertinetto
            ├── test.txt
            ├── train.txt
            ├── val.txt
```
Firstly, obtain the image data in the above format, classify the images according to the `CSV file/txt` file, then generate data for the train image data, and generate a new pickle file through/Data Augmentation/PickleTestpy. Finally, perform directory classification in the following format:

```
├── dataset
    ├── mini-imagenet
        ├── compacted_datasets
            ├── mini_imagenet_test.pickle   
            ├── mini_imagenet_train.pickle  
            ├── mini_imagenet_val.pickle
    ├── cifar-fs
        ├── compacted_datasets
            ├── cifar-fs_tes.pickle
            ├── cifar-fs_train.pickle
            ├── cifar-fs_val.pickle
```
    
        
for your convenience, youcan download the datasets diretly from links on the lieft
1. please download the dataset miniImageNet/cifar-fs like the form of （train/val/test// CSV.）
2. please begin the first data generation
3. train
