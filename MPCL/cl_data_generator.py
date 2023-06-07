import numpy as np
import torch.nn as nn
import torch

class Data_Generator(nn.Module):
    def __init__(self, ld = 0.8, p_length=600, n_length=1000):
        super(Data_Generator,self).__init__()
        self.p_length = p_length
        self.n_length = n_length
        self.ld = ld

    def cutmix_query(self, query,length):
        w = query.size(1)
        num = query.size(0)
        rand_index = torch.randperm(num)
        copy_query = query
        
        
        x = np.random.randint(w)
        x1 = np.clip(x-length//2, 0, w)
        x2 = np.clip(x+length//2, 0, w)

        query[:, x1:x2] = copy_query[rand_index, x1:x2]
        return query
    
    def pip_query(self, query,ld):
        num = query.size(0)
        rand_index = torch.randperm(num)
        noise_query = ld*query + (1-ld)*query[rand_index,:]
        return noise_query

    def __call__(self, query):
        anchor_query = query
        positive_query = self.cutmix_query(query, self.p_length)
        negative_query = self.cutmix_query(query, self.n_length)
        return anchor_query,positive_query,negative_query

class Data_Generator2(nn.Module):
    def __init__(self, p_ld = 0.6, n_ld = 0.4, p_length=600, n_length=1000):
        super(Data_Generator2,self).__init__()
        self.p_length = p_length
        self.n_length = n_length
        self.p_ld = p_ld
        self.n_ld = n_ld

    def cutmix_query(self, query,length):
        w = query.size(1)
        num = query.size(0)
        rand_index = torch.randperm(num)
        copy_query = query
        
        
        x = np.random.randint(w)
        x1 = np.clip(x-length//2, 0, w)
        x2 = np.clip(x+length//2, 0, w)

        query[:, x1:x2] = copy_query[rand_index, x1:x2]
        return query
    
    def pip_query(self, query,ld):
        num = query.size(0)
        rand_index = torch.randperm(num)
        noise_query = ld*query + (1-ld)*query[rand_index,:]
        return noise_query

    def __call__(self, proto):
        anchor_query = proto
        positive_query = self.pip_query(proto, self.p_ld)
        negative_query = self.pip_query(proto, self.n_ld)
        return anchor_query,positive_query,negative_query

class Data_Generator3(nn.Module):
    def __init__(self, p_ld = 0.8, n_ld = 0.2, p_length=600, n_length=1000):
        super(Data_Generator3,self).__init__()
        self.p_length = p_length
        self.n_length = n_length
        self.p_ld = p_ld
        self.n_ld = n_ld

    def cutmix_query(self, query,length):
        w = query.size(1)
        num = query.size(0)
        rand_index = torch.randperm(num)
        copy_query = query
        
        
        x = np.random.randint(w)
        x1 = np.clip(x-length//2, 0, w)
        x2 = np.clip(x+length//2, 0, w)

        query[:, x1:x2] = copy_query[rand_index, x1:x2]
        return query
    
    def pip_query(self, query,ld):
        num = query.size(0)
        rand_index = torch.randperm(num)
        noise_query = ld*query + (1-ld)*query[rand_index,:]
        return noise_query

    def __call__(self,proto, query):
        anchor_query = proto
        positive_query = self.pip_query(query, self.p_ld)
        negative_query = self.pip_query(query, self.n_ld)
        return anchor_query,positive_query.detach(),negative_query.detach()

class Dy_Data_Generator(nn.Module):
    def __init__(self, p_length=600, n_length=1000):
        super(Dy_Data_Generator,self).__init__()
        self.p_length = p_length
        self.n_length = n_length
        

    def cutmix_query(self, query,length):
        w = query.size(1)
        num = query.size(0)
        rand_index = torch.randperm(num)
        copy_query = query
        
        
        x = np.random.randint(w)
        x1 = np.clip(x-length//2, 0, w)
        x2 = np.clip(x+length//2, 0, w)

        query[:, x1:x2] = copy_query[rand_index, x1:x2]
        return query
    
    def pip_query(self, query,ld):
        num = query.size(0)
        rand_index = torch.randperm(num)
        noise_query = ld.view(num,1)*query + (1-ld.view(num,1))*query[rand_index,:]
        return noise_query

    def __call__(self, proto,p_ld, n_ld):
        anchor_query = proto
        p_ld = torch.clamp(p_ld,0.6,1)#original setting:0.6
        n_ld = 1.0-p_ld
        # p_ld, _ = torch.max(ax, dim=1)
        # n_ld, _ = torch.min(ax,dim=1)
        
        positive_query = self.pip_query(proto,p_ld)
        negative_query = self.pip_query(proto,n_ld)
        return anchor_query,positive_query,negative_query

if __name__=='__main__':
    proto = torch.randn([5,1600]).cuda()
    data_generator = Data_Generator().cuda()
    x1,x2,x3 = data_generator(proto)
    print(x1.size())

