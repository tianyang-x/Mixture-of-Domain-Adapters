import typing
from torch.utils.data import Dataset
import torch
import pandas as pd
import json
import random


class TaskDataset(Dataset):
    """
    This corresponds to the task datasets that are labelled.
    """
    def __init__(
        self, data_path: str, label2id=None, few_shot=None, *args, **kwargs
    ):
        self.sentences = []
        self.labels = []
        if label2id is None:
            self.label2id = {}
        else:
            self.label2id = label2id
        with open(data_path, "r") as f:
            for row in f:
                data = json.loads(row)
                self.sentences.append(data['text'])
                label = data['label']
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)
                self.labels.append(self.label2id[label])
                
        if few_shot is not None:
            new_results = self.preprocess_function_k_shot(
                {'sentences': self.sentences, 'label': self.labels}, few_shot
            )
            self.sentences = new_results['sentences']
            self.labels = new_results['label']

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return {
            'text': self.sentences[item],
            'label': self.labels[item],
        }
        
    def collate_fn(self, batch):
        return {
            'text': [x['text'] for x in batch],
            'label': [x['label'] for x in batch],
        }

    @staticmethod
    def preprocess_function_k_shot(examples, few_shot):
        random_indices = list(range(0, len(examples["label"])))
        random.shuffle(random_indices)

        new_examples = {}
        for key in examples.keys():
            new_examples[key] = []
        label_count = {}

        for index in random_indices:      
            label = examples['label'][index]
            if label not in label_count:
                label_count[label] = 0

            if label_count[label] < few_shot:
                for key in examples.keys():
                    new_examples[key].append(examples[key][index])
                label_count[label] += 1
        
        print('k-shot selection done!!')
        
        return new_examples
