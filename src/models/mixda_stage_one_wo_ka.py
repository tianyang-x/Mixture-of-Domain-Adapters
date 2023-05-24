from argparse import ArgumentParser
from datetime import datetime
from datasets import load_dataset
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
import torch
from itertools import chain
from src.data.combined_dataset import CombinedDataset

from .transformers import RobertaForMaskedLM, RobertaConfig
from transformers.adapters.configuration import PfeifferConfig
from transformers import DataCollatorForLanguageModeling
import os
import datasets
from datasets import Dataset


class MixDAAdapterAblation(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dirpath", type=str, default="models/mod_stage_one_wo_ka"
        )
        parser.add_argument(
            "--knowledge_data_path",
            type=str,
            default="datasets/dataset/train.json",
        )
        parser.add_argument("--batch_size", type=int, default=20)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--max_length", type=int, default=32)
        parser.add_argument("--total_num_updates", type=int, default=30)
        parser.add_argument("--warmup_updates", type=int, default=5)
        parser.add_argument("--knowledge_loss_factor", type=float, default=0.5)

        parser.add_argument("--model_name", type=str, default="roberta-large")
        parser.add_argument(
            "--model_checkpoint",
            type=str,
            default="",
        )
        parser.add_argument("--dataset_size_limit", type=int, default=-1)
        
        return parser
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.hparams.model_name)
        adapter_config = PfeifferConfig(
            ln_after=False,
            ln_before=False,
            mh_adapter=False,
            output_adapter=True,
            adapter_residual_before_ln=False,
            non_linearity='swish',
            original_ln_after=True,
            original_ln_before=True,
            reduction_factor=16,
            residual_before_ln=True
        )
        model_config = RobertaConfig.from_pretrained(
            self.hparams.model_name,
            output_original=False,
            enable_new_ka=False,
            enable_old_ka=False,
        )
        self.model = RobertaForMaskedLM.from_pretrained(self.hparams.model_name, config=model_config).to(self.device)
        self.model.add_adapter('domain', 'pfeiffer', adapter_config)
        self.model.train_adapter('domain', 'pfeiffer')
        
        # prepare loss functions
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        
        self.model_filename = self.hparams.dirpath + '/model' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        
    def train_dataloader(self, shuffle=False):
        def map_dataset(x):
            if len(x['targets']) != 0:
                x['text'] = x['prompt'].format(*x['targets'])
            else:
                x['text'] = x['prompt']
            return x
        if not hasattr(self, 'knowledge_dataset'):
            self.knowledge_dataset = \
                load_dataset('json', data_files=self.hparams.knowledge_data_path)['train']
            self.knowledge_dataset = \
                self.knowledge_dataset.map(map_dataset)
            self.knowledge_dataset = self.knowledge_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=self.knowledge_dataset.column_names
            )
            self.knowledge_dataset = self.knowledge_dataset.map(
                self.group_texts,
                batched=True,
            )
        if not hasattr(self, 'sample_dataset'):
            sample_dataset_raw = load_dataset(
                'wikipedia', '20220301.en'
            )['train'][:len(self.knowledge_dataset)]
            sample_dataset_raw = Dataset.from_dict(sample_dataset_raw)
            tokenized_datasets = sample_dataset_raw.map(
                self.tokenize_function,
                batched=True,
                remove_columns=sample_dataset_raw.column_names,
            )
            tokenized_datasets = tokenized_datasets.map(
                self.group_texts,
                batched=True,
            )
            self.sample_dataset = tokenized_datasets
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)
        self.train_dataset = CombinedDataset(self.knowledge_dataset, self.sample_dataset, collate_2=data_collator, collate_1=data_collator)
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=shuffle,
            num_workers=16,
            drop_last=True
        )
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    
    def training_step(self, batch, batch_idx=None):
        knowledge_data, sample_data = batch
        # sample_layers = [self.hparams.mlp_module.format(layer) for layer in self.hparams.layers]
        
        sample_logits = self(**sample_data)
        sample_loss = sample_logits.loss
        
        # Compute knowledge loss
        knowledge_loss = self(**knowledge_data).loss
        
        # Compute total loss
        loss = self.hparams.knowledge_loss_factor * knowledge_loss + sample_loss
        
        self.log('knowledge_loss', knowledge_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('sample_loss', sample_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay= self.hparams.weight_decay)

        return optimizer
        
    def on_train_epoch_end(self) -> None:
        if not os.path.isdir(self.model_filename):
            os.mkdir(self.model_filename)
        self.model.save_adapter(self.model_filename, 'domain')
        
    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], return_special_tokens_mask=True)
    
    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        max_seq_length = self.tokenizer.model_max_length
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
