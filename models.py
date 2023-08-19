## Author: Han Wu (han.wu@sydney.edu.au)

from embedding import *
from d_embedding import *
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        residual = x
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask==0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class transformer_block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        print('attn_drop: {}, drop: {}, drop path rate: {}'.format(attn_drop, drop, drop_path))
        self.out_dim = out_dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0.2, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        size = x.shape
        x, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = self.drop_path(self.mlp(self.norm2(x)))

        return x.mean(dim=1).view(size[0], 1, 1, self.out_dim), x.view(-1, self.out_dim)

class ContextLearner(nn.Module):
    def __init__(self, num_symbols, embed, embed_dim, few, batch_size,
                 dim, num_heads, qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(ContextLearner, self).__init__()
        self.num_symbols = num_symbols
        self.embed_dim = embed_dim
        self.few = few
        self.batch_size = batch_size
        self.symbol2emb = nn.Embedding(num_symbols + 1, self.embed_dim, padding_idx=self.num_symbols)
        self.symbol2emb.weight.data.copy_(torch.from_numpy(embed))
        self.symbol2emb.weight.requires_grad = True   
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0.2, proj_drop=drop)
        self.fc = nn.Linear(dim, dim//2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, connections, mask):
        relations = connections[:, :, :, 0]
        entities = connections[:, :, :, 1]
        
        rel_embeds = self.symbol2emb(relations)                          
        entity_embeds = self.symbol2emb(entities)                        
        
        neighbor_embeds = torch.cat((rel_embeds, entity_embeds), dim=3).reshape(-1, 100, self.embed_dim*2) 
        mask = mask.reshape(-1, 100, 100)

        neighbor_embeds, attn = self.attn(self.norm1(neighbor_embeds), mask)
        neighbor_embeds = self.drop_path(neighbor_embeds) 

        weighted_context = torch.bmm(attn.mean(dim=2), neighbor_embeds.squeeze(1))
        weighted_context = self.drop_path(self.fc(self.norm2(weighted_context)))
        return weighted_context.squeeze(1)    

class EmbeddingLearner(nn.Module):
    def __init__(self):
        super(EmbeddingLearner, self).__init__()

    def forward(self, h, t, r, pos_num, norm_transfer):
        # TransD
        h_transfer, r_transfer, t_transfer = norm_transfer
        h_transfer = h_transfer[:,:1,:,:]
        r_transfer = r_transfer[:,:1,:,:]
        t_transfer = t_transfer[:,:1,:,:]	
        h = h + torch.sum(h * h_transfer, -1, True) * r_transfer
        t = t + torch.sum(t * t_transfer, -1, True) * r_transfer
        
        score = -torch.norm(h + r - t, 2, -1).squeeze(2)
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score

class Hire(nn.Module):
    def __init__(self, dataset, parameter, num_symbols, embed=None):
        super(Hire, self).__init__()
        self.device = parameter['device']
        self.beta = parameter['beta']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.few = parameter['few']
        self.batch_size = parameter['batch_size']
        self.max_neighbor = parameter['max_neighbor']
        self.embedding = Embedding(dataset, parameter) 
        self.loss_func = nn.MarginRankingLoss(self.margin)
        self.rel_q_sharing = dict()
        self.embedding_learner = EmbeddingLearner()
        self.d_embedding = D_Embedding(dataset, parameter)
        self.d_norm = None
        
        if parameter['dataset'] == 'Wiki-One':
            self.context_learner = ContextLearner(num_symbols, embed, self.embed_dim, self.few, self.batch_size, dim=100, num_heads=1, drop=0.2, drop_path=0.2)
            self.relation_learner = transformer_block(dim=100, out_dim=50, num_heads=1, drop=0.2, drop_path=0.2)
        elif parameter['dataset'] == 'NELL-One':
            self.context_learner = ContextLearner(num_symbols, embed, self.embed_dim, self.few, self.batch_size, dim=200, num_heads=1, drop=0.2, drop_path=0.2)
            self.relation_learner = transformer_block(dim=200, out_dim=100, num_heads=1, drop=0.2, drop_path=0.2)
        else:
            print("Wrong dataset name")

    def split_concat(self, positive, negative):
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], 1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], 1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2
    
    def build_context(self, meta):
        left_connections = torch.stack([meta[few_id][0] for few_id in range(self.few)], dim=1)  
        right_connections = torch.stack([meta[few_id][2] for few_id in range(self.few)], dim=1) 
        left_degrees = torch.stack([meta[few_id][1] for few_id in range(self.few)], dim=1).reshape(-1)     
        right_degrees = torch.stack([meta[few_id][3] for few_id in range(self.few)], dim=1).reshape(-1)
        
        left_digits = torch.zeros(self.batch_size*self.few, self.max_neighbor).to(self.device)
        right_digits = torch.zeros(self.batch_size*self.few, self.max_neighbor).to(self.device)
        for i in range(self.batch_size*self.few):
            left_digits[i, :left_degrees[i]] = 1
            right_digits[i, :right_degrees[i]] = 1
        left_digits = left_digits.reshape(-1, self.few, self.max_neighbor)
        right_digits = right_digits.reshape(-1, self.few, self.max_neighbor)
        
        connections = torch.cat((left_connections, right_connections), dim=2)         
        mask = torch.cat((left_digits, right_digits), dim=2)                          
        mask_matrix = mask.reshape(-1, self.max_neighbor*2).unsqueeze(2)
        mask = torch.bmm(mask_matrix, mask_matrix.transpose(1,2))
        
        return connections, mask.reshape(self.batch_size, self.few, self.max_neighbor*2, self.max_neighbor*2)

    def forward(self, task, iseval=False, curr_rel='', support_meta=None, support_negative_meta=None):
        support, support_negative, query, negative = [self.embedding(t) for t in task]
        transfer_vector = self.d_embedding(task[0])
        
        batch_size = support.shape[0]
        few = support.shape[1]              # num of few
        num_sn = support_negative.shape[1]  # num of support negative
        num_q = query.shape[1]              # num of query
        num_n = negative.shape[1]           # num of query negative
        
        # positive and negative views
        if not iseval:
            positive_connections, positive_mask = self.build_context(support_meta)
            negative_connection_mask = [self.build_context(support_nn) for support_nn in support_negative_meta]

            positive_context = self.context_learner(positive_connections, positive_mask)
            negative_context = [self.context_learner(negative_cm[0], negative_cm[1]) for negative_cm in negative_connection_mask]
        else:
            positive_context, negative_context = None, None

        rel, support_emb = self.relation_learner(support.contiguous().view(batch_size, few, -1))  
        rel.retain_grad()
        transfer_vector[0].retain_grad()
        transfer_vector[1].retain_grad()
        transfer_vector[2].retain_grad()
        
        # relation for support
        rel_s = rel.expand(-1, few+num_sn, -1, -1)

        if iseval and curr_rel != '' and curr_rel in self.rel_q_sharing.keys():
            rel_q = self.rel_q_sharing[curr_rel]
        else:
            sup_neg_e1, sup_neg_e2 = self.split_concat(support, support_negative)
            p_score, n_score = self.embedding_learner(sup_neg_e1, sup_neg_e2, rel_s, few, transfer_vector)
            y = torch.Tensor([1]).to(self.device)
            self.zero_grad()
            loss = self.loss_func(p_score, n_score, y)
            loss.backward(retain_graph=True)

            rel_grad_meta = rel.grad
            rel_q = rel - self.beta * rel_grad_meta
            h_grad_meta = transfer_vector[0].grad
            r_grad_meta = transfer_vector[1].grad
            t_grad_meta = transfer_vector[2].grad
            norm_h = transfer_vector[0] - self.beta * h_grad_meta
            norm_r = transfer_vector[1] - self.beta * r_grad_meta
            norm_t = transfer_vector[2] - self.beta * t_grad_meta
            norm_transfer = (norm_h, norm_r, norm_t)

            self.rel_q_sharing[curr_rel] = rel_q
            self.d_norm = (transfer_vector[0].mean(0).unsqueeze(0), transfer_vector[1].mean(0).unsqueeze(0), transfer_vector[2].mean(0).unsqueeze(0))

        rel_q = rel_q.expand(-1, num_q + num_n, -1, -1)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative) 
        
        if iseval:
            norm_transfer = self.d_norm
        p_score, n_score = self.embedding_learner(que_neg_e1, que_neg_e2, rel_q, num_q, norm_transfer)

        return p_score, n_score, positive_context, negative_context, support_emb

