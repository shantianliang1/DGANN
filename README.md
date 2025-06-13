# Few shot Learning Image Classification Based on Double-ended Graph Augmentation Neural Networks
## Abastract
Few-Shot Learning (FSL) aims to achieve effective classification of new categories using limited labeled samples. However, existing approaches face two critical challenges: meta-learning-based methods are prone to overfitting, while Graph Neural Network (GNN)-based methods underutilize image features, particularly showing poor performance on small-sized images. To address these issues, this paper proposes a Dual-End Graph Augmentation Neural Network (DGANN) that synergistically optimizes few-shot image classification through innovative dual-end augmentation strategies and graph reasoning mechanisms.The core innovations of DGANN include: (1) A dual-end augmentation module comprising front-end data generation and back-end data enhancement stages, systematically expanding feature space coverage through geometric transformations and noise injection strategies; (2) A hierarchical graph reasoning module integrating ENC feature embedding, SubSample/UpSample node reasoning, and ResMLP residual connections for deep feature learning on graph-structured data; (3) The first deep integration of data augmentation techniques with GNN architectures, specifically optimized for processing limitations of small-sized image datasets.Extensive experiments on three benchmark datasets validate the method's effectiveness. On miniImageNet, classification accuracies for 5-way 1-shot, 5-way 5-shot, and 10-way 5-shot tasks improved by 4.5%, 2%, and 14% respectively. On CIFAR-FS, the method breaks through traditional GNN performance bottlenecks on small-sized images, achieving 1.1% and 2% improvements for 5-way 1-shot and 5-way 5-shot tasks, and reaching 63% accuracy on 10-way 5-shot tasks. On CUB200, classification accuracies reached 75.86% and 89.40%, with an additional 1% improvement when incorporating ResMLP and approximately 4× acceleration in convergence speed. Experimental results demonstrate that DGANN maintains high classification accuracy while significantly improving training efficiency and model generalization capability, providing a new technical pathway for few-shot learning.

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
author={Qiguang Zhu, Xiaotian Ying, Yanying Zhu, Weidong Chen},
journal={Computer Vision and Image Understanding}
}
```
