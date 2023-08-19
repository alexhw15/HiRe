## Author: Han Wu (han.wu@sydney.edu.au)

import torch
import torch.nn as nn


class D_Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(D_Embedding, self).__init__()
        self.device = parameter['device']
        self.embed_dim = parameter['embed_dim']
        self.ent2id = dataset['ent2id']
        self.rel2id = dataset['rel2id']
        num_ent = len(self.ent2id)
        num_rel = len(self.rel2id)
        
        self.ent_transfer = nn.Embedding(num_ent, self.embed_dim)
        self.rel_transfer = nn.Embedding(num_rel, self.embed_dim)
        nn.init.xavier_uniform_(self.ent_transfer.weight)
        nn.init.xavier_uniform_(self.rel_transfer.weight)

    def forward(self, triples):
        h_id = [[[self.ent2id[t[0]]] for t in batch] for batch in triples]
        r_id = [[[self.rel2id[t[1]]] for t in batch] for batch in triples]
        t_id = [[[self.ent2id[t[2]]] for t in batch] for batch in triples]
        h_id = torch.LongTensor(h_id).to(self.device)
        r_id = torch.LongTensor(r_id).to(self.device)
        t_id = torch.LongTensor(t_id).to(self.device)
        h_transfer = self.ent_transfer(h_id)
        r_transfer = self.rel_transfer(r_id)
        t_transfer = self.ent_transfer(t_id)
        return (h_transfer, r_transfer, t_transfer)
