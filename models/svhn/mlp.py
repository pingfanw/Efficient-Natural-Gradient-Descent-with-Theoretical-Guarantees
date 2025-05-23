import torch.nn as nn
import torch.nn.functional as F
__all__ = ['mlp']

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(2,16)  
        self.fc2 = nn.Linear(16,32)  
        self.fc3 = nn.Linear(32,16)  
        self.fc4 = nn.Linear(16,2)  

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)  
        x = self.fc2(x)
        x = F.relu(x)  
        x = self.fc3(x)
        x = F.relu(x)  
        x = self.fc4(x)
        return x
    
def mlp(**kwargs):
    model = MLP(**kwargs)
    return model
