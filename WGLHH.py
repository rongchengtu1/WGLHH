from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
import os
import numpy as np
import pickle
from datetime import datetime
import argparse

import utils.DataProcessing as dp
import utils.calc_hr as CalcHR
import time
import utils.CNN_model as CNN_model

def LoadLabel(label_path, item_path, DATA_DIR):
    labels = np.loadtxt(os.path.join(DATA_DIR, label_path), dtype=np.float32)
    items = np.loadtxt(os.path.join(DATA_DIR, item_path), dtype=int) - 1
    label = labels[items]

    return torch.from_numpy(label)

def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S

def CreateModel(model_name, bit):
    if model_name == 'vgg16':
        vgg11 = models.vgg16(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg11, model_name, bit)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        cnn_model = CNN_model.cnn_model(alexnet, model_name, bit)
    cnn_model = cnn_model.cuda()
    return cnn_model

def GenerateCode(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    f = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        _, output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        f[data_ind.numpy(), :] = output.cpu().data
    return B, f

def WGLHH_algo(bit):
    # parameters setting
    DATA_DIR = '/data/home/trc/mat/imagenet'
    LABEL_FILE = 'label_hot.txt'
    IMAGE_FILE = 'images_name.txt'
    DATABASE_FILE = 'database_ind.txt'
    TRAIN_FILE = 'train_ind.txt'
    TEST_FILE = 'test_ind.txt'

    cuda_id = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)

    batch_size = 50
    epochs = 500
    learning_rate = 0.05
    weight_decay = 10 ** -4
    model_name = 'vgg16'
    theta = 2.0
    gamma = 0.001
    alpha = 0.1
    containt0 = 0.0001

    ### ImageNet
    # if bit < 32:
    #     kt = 70
    # elif bit == 64:
    #     kt = 70
    # else:
    #     kt = 70

    ## MS COCO
    if bit < 32:
        kt = 0.1
    elif bit == 64:
        kt = 0.7
    else:
        kt = 0.6

    ### data processing
    transformations = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dset_database = dp.DatasetProcessing(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, DATABASE_FILE, transformations)

    dset_train = dp.DatasetProcessing(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, TRAIN_FILE, transformations)

    dset_test = dp.DatasetProcessing(
        DATA_DIR, IMAGE_FILE, LABEL_FILE, TEST_FILE, transformations)

    num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

    database_loader = DataLoader(dset_database,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=4
                                 )

    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4
                              )

    test_loader = DataLoader(dset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4
                             )

    ### create model
    model = CreateModel(model_name, bit)
    params = [
        {"params": model.hash_layer.parameters(), "lr": learning_rate},
        {"params": model.classifier.parameters(), "lr": learning_rate * 0.1},
        {"params": model.features.parameters(), "lr": learning_rate * 0.1},
    ]
    optimizer = optim.SGD(params, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=401, gamma=0.5, last_epoch=-1)
    test_labels_onehot = LoadLabel(LABEL_FILE, TEST_FILE, DATA_DIR)


    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_lossq = 0.0
        scheduler.step()
        ## training epoch
        for iter, traindata in enumerate(train_loader, 0):
            train_input, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)

            train_label_onehot = train_label.type(torch.FloatTensor)
            train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
            S = CalcSim(train_label_onehot, train_label_onehot)

            model.zero_grad()
            _, train_outputs = model(train_input)

            normal1 = torch.sqrt((train_outputs.pow(2)).sum(1)).view(-1, 1)
            normal_code = train_outputs / (normal1.expand(train_label.size()[0], bit))
            c_sim = normal_code.mm((normal_code).t())
            weight_s = torch.exp(torch.abs((0.5 * c_sim + 0.5 - Variable(S.cuda()))))
            dh = 0.5 * float(bit) * (1 - c_sim)

            lq = (torch.sign(train_outputs) - train_outputs).pow(2)

            pq = torch.exp(-alpha * (dh ** 2))
            lpq = pq * torch.log((theta * pq + containt0) / ((theta - 1) * pq + Variable(S.cuda()) + containt0)) + \
                  Variable(S.cuda()) * torch.log((theta * Variable(S.cuda()) + (1 - Variable(S.cuda())) * 0.1) / (
                    (theta - 1) * Variable(S.cuda()) + (1 - Variable(S.cuda())) * 0.0001 + pq))
            loss = ((Variable(S.cuda()) * kt + 1) * weight_s * lpq).sum() / (
                        train_label.size()[0] * train_label.size()[0]) + gamma * lq.sum() / train_label.size()[0]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_lossq += (lq.sum() / train_label.size()[0]).item()

        if (epoch % 100 == 0)&(epoch != 0):
            model.eval()
            database_labels_onehot = LoadLabel(LABEL_FILE, DATABASE_FILE, DATA_DIR)
            qB, qf = GenerateCode(model, test_loader, num_test, bit)
            dB, df = GenerateCode(model, database_loader, num_database, bit)
            model.train()
            map_n = CalcHR.calc_HammingMap(qB, dB, qf, df, test_labels_onehot.numpy(), database_labels_onehot.numpy(), 2)
            print('map@2:', map_n)
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f][Lossq1: %3.5f]' %
                    (epoch + 1, epochs, epoch_loss / len(train_loader), epoch_lossq / len(train_loader)))
    model.eval()
    database_labels_onehot = LoadLabel(LABEL_FILE, DATABASE_FILE, DATA_DIR)
    qB, qf = GenerateCode(model, test_loader, num_test, bit)
    dB, df = GenerateCode(model, database_loader, num_database, bit)

    map_n = CalcHR.calc_HammingMap(qB, dB, qf, df, test_labels_onehot.numpy(), database_labels_onehot.numpy(), 2)
    print('map@2:', map_n)


if __name__ == '__main__':
    bits = [24, 32, 48, 64]
    for bit in bits:
        WGLHH_algo(bit)

