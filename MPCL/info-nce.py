import torch
import torch.nn as nn
from info_nce import InfoNCE

if __name__=='__main__':
    loss = InfoNCE()
    query = torch.randn([75, 512])
    p_keys = torch.randn([75, 512])
    n_keys = torch.randn([75,512])
    output = loss(query, p_keys, n_keys)
    print(output)