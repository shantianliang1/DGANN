# Enhanced Few-Shot Image Classification via Double-Ended Graph Augmentation Neural Networks
## Abastract
To address the overfitting issue commonly encountered in meta-learning-based Few-Shot Learning (FSL) methods and the inability of Graph Neural Network (GNN)-based FSL methods to fully utilize image data, this paper proposes a GNN model based on a Double-Ended Graph Augmentation Neural Network (DGANN). First, a dual-end augmentation module is designed to enhance the model's ability to extract features and process graph data. Then, a graph reasoning module is introduced, where the enhanced sample features are obtained through the ENC module, and node features are reasoned using the SubSample and UpSample modules. These features are further processed with residual connections through the ResMLP architecture, improving image classification accuracy. The model is evaluated on the miniImageNet and CIFAR-FS benchmark datasets using 5-way 1-shot, 5-way 5-shot, and 10-way 5-shot classification tasks, as well as on the CUB200 dataset with 5-way 1-shot and 5-way 5-shot classification tasks. Additionally, the convergence speed of the ResMLP architecture is analyzed. For the miniImageNet dataset, classification accuracy increased by 4.5%, 2%, and 14% for the three tasks, respectively. For the CIFAR-FS dataset, DGANN achieved better classification performance on small-sized images compared to traditional GNN models, improving 5-way 1-shot and 5-way 5-shot classification accuracy by 1.1% and 2%, respectively, and achieving 63% accuracy in the 10-way 5-shot task. On the CUB200 dataset, the classification accuracy reached 75.86% and 89.40%, with a 1% improvement when combined with ResMLP, and the convergence speed was accelerated by approximately four times. Experimental results demonstrate that the DGANN model, which incorporates image augmentation techniques, shows higher accuracy and better generalization performance in traditional classification tasks.

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
Firstly, obtain the image data in the above format, classify the images according to the `CSV file/txt` file, then generate data for the train image data, and generate a new pickle file through `/DataAugmentation/pickleTest.py`. Finally, perform directory classification in the following format:

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
    
## Train
`argSettings.py`  is a list of parameters during training, which you need to review carefully and set the corresponding options according to the training you need.

## Eval
Evaluate the trained model by using `eval.py `

## Result
### checkpoints directory is：
`./asset/checkpoints`
### logs directory is：
`./asset/logs`

You can use TensorBoard to see detailed training trends

## Citation
```
@article{Ying2025Gnn,
title={Few shot Learning Image Classification Based on Double-ended Graph Augmentation Neural Networks},
author={Qiguang Zhu, Xiaotian Ying, Xi Lv, Weidong Chen},
journal={Computer Vision and Image Understanding}
}
```
