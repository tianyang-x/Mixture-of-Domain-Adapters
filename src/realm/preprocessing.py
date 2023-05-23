'''
Collect the embeddings of Wikipedia into a numpy array.
'''

from token import tok_name
from transformers import RobertaTokenizerFast, RobertaModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Any
from argparse import ArgumentParser
from datasets import Dataset
from pytorch_lightning.strategies.ddp import DDPStrategy

class EmbeddingCollector(pl.LightningModule):
    def __init__(self, bts, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.roberta = RobertaModel.from_pretrained('roberta-base').eval()
        self.tok = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.batch_size = 32
        self.indices = []
        self.results = []
        self.bts = bts
        
    def val_dataloader(self):
        sample_dataset_raw = load_dataset('wikipedia', '20220301.en')['train']
        ds_len = len(sample_dataset_raw) // 4
        sample_dataset_raw = sample_dataset_raw[ds_len * self.bts : ds_len * (self.bts + 1)]
        sample_dataset_raw = Dataset.from_dict(sample_dataset_raw)
        sample_dataset_raw = sample_dataset_raw.add_column('raw_id', range(len(sample_dataset_raw)))
        self.dl = DataLoader(sample_dataset_raw, batch_size=self.batch_size, num_workers=0,)
        for i in range(torch.cuda.device_count()):
            self.results.append(np.ndarray((ds_len, 768), dtype=np.float32))
            self.indices.append(np.ndarray((ds_len), dtype=np.int32))
        return self.dl
    
    def validation_step(self, batch, batch_idx):
        batch, raw_ids = batch['text'], batch['raw_id']
        batch = self.tok(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.roberta(**batch)
        start_idx = batch_idx * self.batch_size
        for id, raw_id in enumerate(raw_ids):
            dev_id = int(str(self.device)[-1])
            indices, results = self.indices[dev_id], self.results[dev_id]
            self.indices[dev_id][start_idx] = raw_id.cpu().numpy()
            o = output.last_hidden_state[id, 0].unsqueeze(0).cpu().numpy()
            self.results[dev_id][start_idx] = o
            start_idx += 1
        return {'loss': torch.tensor(0.0)}
    
    def validation_epoch_end(self, outputs):
        dev_id = int(str(self.device)[-1])
        np.save('wiki_cls_indices' + str(self.bts), self.indices[dev_id])
        np.save('wiki_cls_results' + str(self.bts), self.results[dev_id])

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch', type=int, default=0)
    args, _ = parser.parse_known_args()
    trainer = pl.Trainer.from_argparse_args(
        args, 
        strategy = DDPStrategy(find_unused_parameters=False),
    )
    model = EmbeddingCollector(bts=args.batch)
    trainer.validate(model)
    