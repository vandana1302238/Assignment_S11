from resnet18_model import ResNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
from torchsummary import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
from utils import *
from utils import trainer, plot_misclassified_images, plot_grad_cam_images


data_path = './data'
batch_size = 512
trainloader,testloader, classes = dataloader(data_path, batch_size)

batch_data, batch_label = next(iter(testloader))
plot_sample_data(testloader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)

summary(model, input_size=(3, 32, 32))

optimizer = optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

get_lr(optimizer)
lr_calc(model, trainloader, optimizer, criterion)

EPOCHS = 20
scheduler = OneCycleLR(
        optimizer,
        max_lr= 4.79E-02,
        steps_per_epoch=len(trainloader),
        epochs=EPOCHS,
        pct_start=5/EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )

trainer = trainer(model, device, optimizer,scheduler)

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    trainer.train(trainloader)
    trainer.test(testloader)
    scheduler.step()

trainer.visualize_graphs()
plot_misclassified_images(model, testloader, classes, device)
plot_grad_cam_images(model, testloader, classes, device)