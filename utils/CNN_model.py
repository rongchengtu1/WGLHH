import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch

class cnn_model(nn.Module):
    def __init__(self, original_model, model_name, bit):
        super(cnn_model, self).__init__()
        if model_name == 'vgg16':
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            cl1.weight = original_model.classifier[0].weight
            cl1.bias = original_model.classifier[0].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[3].weight
            cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                # nn.Dropout(),
            )
            self.hash_layer = nn.Sequential(
                nn.Linear(4096, bit),
                nn.Tanh()
            )
            self.model_name = 'vgg16'
        if model_name == 'alexnet':
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl1.weight = original_model.classifier[1].weight
            cl1.bias = original_model.classifier[1].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[4].weight
            cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
            )
            self.hash_layer = nn.Sequential(
                nn.Linear(4096, bit),
                nn.Tanh()
            )
            self.model_name = 'alexnet'


    def forward(self, x):
        f = self.features(x)
        if self.model_name == 'vgg16':
            f = f.view(f.size(0), -1)
        if self.model_name == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        y = self.classifier(f)
        code = self.hash_layer(y)
        return y, code

if __name__=="__main__":
    alexnet = models.alexnet(pretrained=True)
    print(alexnet)
    #
