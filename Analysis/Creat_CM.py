"""
coding: utf-8
author: Lu Shiliang
date:2023-09
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

if __name__  == '__main__':
    with open('results2.txt', 'r') as f:
        list1 = []
        for line in f:
            print(type(line))
            l = eval(line)
            list1.append(l)  # 这里用eval将字符串转换为代码来执行

        true_labels = list1[0]
        predict_labels = list1[1]

    # print(true_labels, predict_labels)

    json_path = 'class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    with open('class_indices_TRUENAME.json', "r") as f:
        class_Name = json.load(f)
    classes = []
    for k, i in enumerate(class_indict):
        classes.append(class_Name[i])
    print(classes)

    print('recall-macro:{}'.format(recall_score(true_labels, predict_labels, average='macro')))
    print('recall-weighted:{}'.format(recall_score(true_labels, predict_labels, average='weighted')))
    print('precision-macro:{}'.format(precision_score(true_labels, predict_labels, average='macro')))
    print('precision-weighted:{}'.format(precision_score(true_labels, predict_labels, average='weighted')))
    print('f1_macro:', f1_score(true_labels, predict_labels, average='macro'))
    print('f1_weighted:', f1_score(true_labels, predict_labels, average='weighted'))
    print('Acc:', accuracy_score(true_labels, predict_labels))
    sys.exit()


def Creat_CM(true_labels, predict_labels):
    confusion = confusion_matrix(true_labels, predict_labels, normalize='true')

    # print(sum(confusion[:,1]))
    # sys.exit()

    plt.figure(figsize=(30, 28))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes, fontproperties='Times New Roman', fontsize=8)
    plt.yticks(indices, classes, fontproperties='Times New Roman', fontsize=8)
    plt.colorbar()
    plt.xlabel('Predict label', fontsize=24, family='Times New Roman')
    plt.xticks(rotation=90)
    plt.ylabel('True label', fontsize=24, family='Times New Roman')
    iters = np.reshape([[[i, j] for j in range(confusion.shape[0])] for i in range(confusion.shape[1])],
                       (confusion.size, 2))

    for i, j in iters:
        plt.text(j, i, '{:.2f}'.format(confusion[i, j]), va='center', ha='center', fontproperties='Times New Roman',
                 fontsize=6)

    plt.savefig('Confusion_matrix.png', dpi=500)

