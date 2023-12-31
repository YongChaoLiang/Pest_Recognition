# Pest_Recognition
Supplementary material to the article “Pest recognition with higher accuracy and lower parameters for large-scale datasets based on Biformer”

## Dataset used in experiment
This work used the IP102 dataset downloaded from Kaggle. 
Click [here](https://www.kaggle.com/datasets/rtlmhjbn/ip02-dataset) to download this dataset.

For more information, please turn to [IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_IP102_A_Large-Scale_Benchmark_Dataset_for_Insect_Pest_Recognition_CVPR_2019_paper.html)

<img src="https://github.com/YongChaoLiang/Pest_Recognition/raw/main/figure/Fig2.png" width="600px">

## Brief introduction
Because of the high hardware requirements of deep learning-based pest identification, it is difficult to be deployed in the field in actual agricultural production. This work innovatively uses the Bi-level Routing Attention to reduce model size and improve operation efficiency while maintaining high accuracy, which has positive significance for agricultural production.

<img src="https://github.com/YongChaoLiang/Pest_Recognition/raw/main/figure/Fig1.png" width="450px"><img src="https://github.com/YongChaoLiang/Pest_Recognition/raw/main/figure/Fig3.png" width="350px">

<img src="https://github.com/YongChaoLiang/Pest_Recognition/raw/main/figure/Fig4.png" width="800px">

## Main results
In this work, the training strategy of DeiT was followed. As shown in the table below, our model achieved 75.2% top1 accuarcy with 12.7 M parameters, which is better than others.
|Name.|Parans.|FLOPs|Top1 ACC|F1 - macro|Model|
|--|--|--|--|--|--|
|Resnet50|23.7|4.1 G|63.5|0.552|[GoogleDrive](https://drive.google.com/u/0/uc?id=1U2zEsCJlekyrDjYuC5uUwss7IxKfpqlv&export=download)|
|Deit-S|21.7|4.6 G|62.3|0.536|[GoogleDrive](https://drive.google.com/u/0/uc?id=1yrqHKtBPXnIfJSqc6a0UJ1SMC29h6Xc6&export=download)|
|Conformer-T|23.0|5.3 G|61.2|0.509|[GoogleDrive](https://drive.google.com/u/0/uc?id=1ywC_Ep4k5Vzna0rDlqrqC0WLmCqkPs0n&export=download)|
|Swin-T|27.6|4.3 G|63.1|0.547|[GoogleDrive](https://drive.google.com/u/0/uc?id=1csSw5NhUFkcBpIa8P79j-PQ3aMyIc1gp&export=download)|
|Swin-B|86.8|15.5 G|72.4|0.637|[GoogleDrive](https://drive.google.com/u/0/uc?id=1UzvFEbGbGSZA9u7YvIn2vacx8aijDPMT&export=download)|
|**Biformer-T(this work)**|**12.7**|**2.2 G**|**75.2**|**0.691**|[GoogleDrive](https://drive.google.com/u/0/uc?id=1-uSJXy9IP9UKDeJOQwzw2DDfJMFrdK0n&export=download)|

We used GradCAM to visulize the attention map. As shown in the following figure, Biformer could pinpoint pests with less environmental impact.

<img src="https://github.com/YongChaoLiang/Pest_Recognition/raw/main/figure/Fig6.png" width="700px">
