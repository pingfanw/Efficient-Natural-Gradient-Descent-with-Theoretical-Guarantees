import torch.nn as nn


__all__ = ['alexnet']


class AlexNet(nn.Module):

    def __init__(self,num_classes=10,**kwargs):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=5,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(384,256,kernel_size=3,padding=1,bias=True),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,padding=1,bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.classifier = nn.Linear(256,num_classes)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model
