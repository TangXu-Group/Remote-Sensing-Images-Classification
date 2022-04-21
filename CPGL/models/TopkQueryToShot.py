import torch
import torch.nn as nn

def refined_proto_calculate(refined_emb_s,emb_q,refined_shot,q_labels,train_way):
    num_classes = train_way
    num_query = int(emb_q.shape[0]/num_classes)

    refined_proto = refined_emb_s.reshape(refined_shot, train_way, -1).mean(dim=0)#a (train_way,1600) Tensor
    n = emb_q.shape[0]
    m = refined_proto.shape[0]
    a = emb_q.unsqueeze(1).expand(n,m,-1)
    b = refined_proto.unsqueeze(0).expand(n,m,-1)
    refined_dist = ((a-b)**2).mean(dim=2)# a (query*train_way, train_way) tensor
    
    pred = torch.argmax(-refined_dist,1).cuda()
    gt = torch.argmax(q_labels, 1).cuda()
    
    correct = (pred==gt).sum()
    total = num_query*num_classes
    acc = 1.0*correct.float()/float(total)
    return refined_proto, acc, refined_dist

def top_k_per_class(query,support,A,k):
    num_classes = A.shape[1]
    # n_query = int(query.shape[0]/num_classes)
    gap = num_classes
    length = A.shape[0]
    dim = query.shape[1]
    top_k_indices = []
    top_k_sample = []
    k = k
    
    for i in range(num_classes):
        specific_class_distance = A[i:length:gap,:]
        specific_class_sample = query[i:length:gap,:]
        topk_value, indices = torch.topk(specific_class_distance[:,i],k)
        top_k_indices.append(indices)
        top_k_sample.append(specific_class_sample.index_select(0,indices))

    top_k_sample = (torch.stack(top_k_sample,dim=0)).view(-1,dim)

    length = top_k_sample.shape[0]
    gap=k
    top_k_sample_rearrange = []
    for i in range(gap):
        all_class_round = top_k_sample[i:length:gap,:]
        top_k_sample_rearrange.append(all_class_round)
        
    top_k_sample_rearrange = (torch.stack(top_k_sample_rearrange,dim=0)).view(-1,dim)

    top_k_sample_proto = top_k_sample_rearrange.view(k,num_classes,-1).mean(dim=0)#计算topk个query的原形，维度为(num_classes,dimension)
    # refined_emb_s = torch.cat((support,top_k_sample_rearrange),dim=0)
    refined_emb_s = torch.cat((support,top_k_sample_proto),dim=0)
    return refined_emb_s

# if __name__ == '__main__':
#     query = 15
#     num_classes = 5
#     A = torch.randn([query*num_classes,num_classes])
#     query_sample = torch.randn([query*num_classes,20]) 
#     gap = num_classes
#     length = A.shape[0]
#     print(A)
#     print('#'*50)
#     print(query_sample)
#     print('#'*50)
    
#     top_k_indices=[]
#     top_k_sample = []
#     k=10
#     for i in range(num_classes):
#         specific_class = A[i:length:gap,:]
#         specific_class_sample = query_sample[i:length:gap,:]
#         topk_value, indices = torch.topk(specific_class[:,i],k)
#         top_k_indices.append(indices)
#         top_k_sample.append(specific_class_sample.index_select(0,indices))
#         # print(specific_class)
#         # print(specific_class_sample)
#         # print('*'*50)
#         # print(specific_class[:,i])
#         # print(indices)
#         # print('^'*50)
        
#     print(top_k_indices)
#     print(top_k_sample)
#     top_k_sample = (torch.stack(top_k_sample,dim=0)).view(-1,20)
#     print('*'*50)
#     print(top_k_sample.shape)

#     length = top_k_sample.shape[0]
#     # print(length)
#     gap=k
#     upper = 0
#     top_k_sample_rearrange = []
#     for i in range(gap):
#         all_class_round = top_k_sample[i:length:gap,:]
#         # print(all_class_round)
#         top_k_sample_rearrange.append(all_class_round)
        
#     top_k_sample_rearrange = (torch.stack(top_k_sample_rearrange,dim=0)).view(-1,20)
#     print(top_k_sample_rearrange.size())

    
