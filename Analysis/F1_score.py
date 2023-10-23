"""
coding: utf-8
author: Lu Shiliang
date:2023-09
"""

import json
import os
import sys

import numpy as np
import torch
from torchvision import transforms

from tqdm import tqdm
import BiFormer.models.biformer as biformer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from Dataset.utils import read_data
from Dataset.MyDataset import ImageDataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = biformer.biformer_tiny(pretrained=False, nm_class=102)
    weight_dict = torch.load('../Output/Real/checkpoint.pth')
    model.load_state_dict(weight_dict["model"], strict=True)


    model.to('cuda')
    model.eval()
    datadict = read_data('E:\Code\Pytorch_gpu\Data\pest\classification', 3)
    img_size = 224
    batch_size = 12
    data_transform = {
        "test": transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    test_dataset = ImageDataset(images_path=datadict['test'][0],
                                images_class=datadict['test'][1],
                                transform=data_transform["test"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=4
                                              )
    predict_labels = []
    true_labels = []
    # sample_num = 0
    data_loader = tqdm(test_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        image, label = data
        for i in range(len(label)):
            true_labels.append(label[i].cpu().numpy())

        # sample_num += image.shape[0]

        pred = model(image.to('cuda'))
        pred_classes = torch.max(pred, dim=1)[1].cpu()
        for i in range(len(pred_classes)):
            predict_labels.append(pred_classes[i].cpu().numpy())

        macro = f1_score(true_labels, predict_labels, average='macro')
        weighted = f1_score(true_labels, predict_labels, average='weighted')
        acc = accuracy_score(true_labels, predict_labels)

        data_loader.desc = "[testing. F1-macro {:.3f},F1-weighted: {:.3f}, Acc: {:.3f}".format(macro, weighted, acc)

    print('macro:', f1_score(true_labels, predict_labels, average='macro'))
    print('weighted:', f1_score(true_labels, predict_labels, average='weighted'))
    print('Acc:', accuracy_score(true_labels, predict_labels))

    for i in range(len(true_labels)):
        true_labels[i] = int(true_labels[i])
        predict_labels[i] = int(predict_labels[i])

    with open('results2.txt','w') as f:
        f.write(str(true_labels))
        f.write('\n')
        f.write(str(predict_labels))

    json_path = 'class_indices.json'
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    with open('class_indices_TRUENAME.json', "r") as f:
        class_Name = json.load(f)
    classes = []
    for k, i in enumerate(class_indict):
        classes.append(class_Name[i])
    print(classes)

    confusion = confusion_matrix(true_labels, predict_labels, normalize='true')
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
        plt.text(j, i, '{:.3f}'.format(confusion[i, j]), va='center', ha='center', fontproperties='Times New Roman',
                fontsize=5)

    plt.savefig('Confusion_matrix.png', dpi=300)

    sys.exit()