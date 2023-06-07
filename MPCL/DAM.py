import torch.nn as nn
import torch

class Dynamic_Attention_Module(nn.Module):
    def __init__(self, in_channels, rate=5):
        super(Dynamic_Attention_Module, self).__init__()
        self.dam = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace = True),
            nn.Linear(int(in_channels/rate), 2)
        )

        

    def forward(self,x):
        x = torch.flatten(x,0)
        x = self.dam(x)
        pi, pi_2 = torch.softmax(x,0)
        return pi, pi_2

if __name__=='__main__':
    x = torch.randn([75, 5])
    # x = torch.flatten(x,0)
    # print(x.size())
    # print(x.size(1))
    dam = Dynamic_Attention_Module(5*5*15)
    pi, pi_2 = dam(x)
    print(pi)
    print(pi_2)