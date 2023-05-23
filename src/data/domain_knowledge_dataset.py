import random
import typing
from torch.utils.data import Dataset
import torch
import pandas as pd
import json


class DomainKnowledgeDataset(Dataset):
    """
    Dataset for unlabelled domain knowledge.
    """
    def __init__(
        self, data_path: str, size: typing.Optional[int] = None, randomize=False, *args, **kwargs
    ):
        print("Loading Knowledge dataset")
        
        with open(data_path, "r") as f:
            data = json.load(f)
        if randomize:
            random.shuffle(data)
        if size is not None:
            data = data[:size]
        self.data = self.preprocess(data)

        print(f"Loaded dataset with {len(self)} elements")
    
    def preprocess(self, data):
        return {
            'id': list(range(len(data))),
            'prompt': [x['prompt'] for x in data],
            'targets': [x['targets'] for x in data],
        }

    def __len__(self):
        return len(self.data['id'])

    def __getitem__(self, item):
        return {
            'id': self.data['id'][item],
            'prompt': self.data['prompt'][item],
            'targets': self.data['targets'][item],
        }
        
    def collate_fn(self, batch):
        return {
            'id': [x['id'] for x in batch],
            'prompt': [x['prompt'] for x in batch],
            'targets': [x['targets'] for x in batch],
        }
