"""
coding: utf-8
author: Lu Shiliang
date:2023-09
"""

import json

from torchvision.transforms import transforms

from Dataset.utils import read_data
import matplotlib.pyplot as plt
from Dataset.MyDataset import ImageDataset


if __name__ == "__main__":
    datadict = read_data('E:\Code\Pytorch_gpu\Data\pest\classification', 3)

    label_distribution_ori = datadict['train'][1]
    # print(len(label_distribution_ori))
    label_distribution = {}
    with open('class_indices_TRUENAME.json', "r") as f:
        class_Name = json.load(f)
    for i in range(102):
        label_distribution[class_Name[str(i)]] = label_distribution_ori.count(i)
    print(label_distribution)
    num_max = 0
    num_min = 999999
    for label in label_distribution.keys():
        num = label_distribution[label]
        if num > num_max:
            num_max = num
            label_max = label
        else:
            if num < num_min:
                num_min = num
                label_min = label

    print('MAX:',label_max, num_max)
    print('MIN:',label_min,num_min)



