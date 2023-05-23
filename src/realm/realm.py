'''
Classes to support REALM.
'''

import torch
from torch import nn
import numpy as np


class RealmRetriever(nn.Module):
    """
    This class retrieves related information from records to support REALM-style information retrieval.
    """
    def __init__(self, doc_records, model_hidden_size=768, query_embed_size=768, doc_hidden_size=768, doc_proj_size=768,
    num_docs=1000, update_emb_freq=5):
        super().__init__()
        # project head for queries
        self.embed_head = nn.Linear(model_hidden_size, doc_hidden_size)
        # project head for documents
        # self.proj_head = nn.Linear(doc_hidden_size, doc_proj_size)
        # document embeddings. to minimize computations, this is not updated unless instructed.
        self.block_emb = [torch.Tensor(doc_records, device='cpu')]
        self.update_emb_freq = update_emb_freq
    
    def forward(self, query, top_k): 
        query_embed = self.embed_head(query)
        # CPU computation. Only used to decide the top k documents.
        batch_scores = torch.einsum(
            "BD,QD->QB", self.block_emb[0], query_embed.to(self.block_emb[0].device))
        _, retrieved_block_ids = torch.topk(batch_scores, top_k, dim=-1)
        retrieved_block_ids = retrieved_block_ids.squeeze()
        return retrieved_block_ids
        