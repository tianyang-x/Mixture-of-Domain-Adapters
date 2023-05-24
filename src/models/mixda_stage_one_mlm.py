from argparse import ArgumentParser
from datetime import datetime
from datasets import load_dataset
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
import torch

from src.data.combined_dataset import CombinedDataset

from .transformers import RobertaForMaskedLM, RobertaConfig, WeightTensor
from ..data.tok_dataset import TokenizedDataset
from .. import utils
import numpy as np
from transformers import DataCollatorForLanguageModeling
from itertools import chain


from ..realm.realm import RealmRetriever


class MixDAStageOneMLM(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dirpath", type=str, default="models/mod_stage_one_mlm"
        )
        parser.add_argument(
            "--knowledge_data_path",
            type=str,
            default="datasets/dataset/train.json",
        )
        parser.add_argument("--batch_size", type=int, default=20)
        # parser.add_argument("--sample_dataset", type=str, default='wikipedia')
        # parser.add_argument("--sample_batch_size", type=int, default=40)
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

        parser.add_argument('--reweighting', default=False, action='store_true')
        parser.add_argument('--layer_reweighting', default=True,action='store_true')
        parser.add_argument('--layers', type=str, default="7")
        parser.add_argument("--dataset_size_limit", type=int, default=-1)
        
        # For specifying adapter & MLP modules
        parser.add_argument("--adapter_module", type=str, default='roberta.encoder.layer.{}.output.kas')
        parser.add_argument("--mlp_module", type=str, default='roberta.encoder.layer.{}.output')
        parser.add_argument("--load_adapters", type=str)
        
        parser.add_argument("--adapter_down_scale", type=float, default=1.0)
        
        parser.add_argument("--realm_record", type=str)
        parser.add_argument("--no_new_knowledge",
                            default=False, action='store_true')
        parser.add_argument("--disable_moe", default=False,
                            action='store_true')
        parser.add_argument("--no_old_knowledge",
                            default=False, action='store_true')


        return parser
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.layers = None if self.hparams.layers is None or self.hparams.layers == '' else [int(x) for x in self.hparams.layers.split(',')]
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.hparams.model_name)
        model_config = RobertaConfig.from_pretrained(
            self.hparams.model_name,
            layers=self.hparams.layers,
            output_original=True,
            enable_new_ka=False,
            enable_old_ka=True,
            adapter_down_scale=self.hparams.adapter_down_scale,
            num_of_kas=1,
            disable_moe=self.hparams.disable_moe,
        )
        self.model = RobertaForMaskedLM.from_pretrained(self.hparams.model_name, config=model_config).to(self.device)
        self.ka_list = None if self.hparams.load_adapters is None or self.hparams.load_adapters == '' else [
            x for x in self.hparams.load_adapters.split(',')]

        if self.ka_list is not None and len(self.ka_list) > 0:
            self.model.load_knowledge_adapter(self.ka_list)
            print('adapter loaded!!!')
            
        # enable REALM
        self.realm = None
        if self.hparams.realm_record is not None:
            doc_records = np.load(self.hparams.realm_record)
            self.realm = RealmRetriever(
                doc_records,
                model_hidden_size=self.model.config.hidden_size,
                query_embed_size=128,
                doc_proj_size=128,
                num_docs=len(doc_records)
            )
            
        # freeze parameters in some layers
        layer_list = []
        for layer in self.hparams.layers:
            layer_list.append('model.' + self.hparams.adapter_module.format(layer))
            layer_list.append('model.roberta.encoder.layer.{}.output.gating'.format(layer))
        # utils.enable_grad_by_module(self.model, layer_list)
        utils.enable_grad_by_module(self, layer_list)
        
        print('not frozen:')
        for name, p in self.named_parameters():
            if p.requires_grad:
                print(name)
        
        # prepare loss functions
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        
        self.model_filename = self.hparams.dirpath + '/model' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + '.pt'
        
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
            )
            self.sample_dataset = TokenizedDataset(
                sample_dataset_raw['train'], 
                self.tokenizer, 
                maxlen=self.model.config.max_position_embeddings
            )
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer)
        self.train_dataset = CombinedDataset(self.knowledge_dataset, self.sample_dataset, collate_1=data_collator)
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
        sample_layers = ['roberta.encoder.layer.{}'.format(
            layer) for layer in self.hparams.layers]
        
        # Compute sample loss
        if not self.hparams.no_old_knowledge:
            if self.realm is not None:
                # Fetch "sample_data" from REALM
                ans_rep = self.model(
                    knowledge_data['input_ids'], output_hidden_states=True).hidden_states[-1][:, 0]
                ids = self.realm(ans_rep, len(knowledge_data))
                ids = ids.flatten().cpu().tolist()
                items = [self.sample_dataset[id] for id in ids]
                sample_data = self.sample_dataset.collate_fn(items)
                for key in sample_data:
                    sample_data[key] = sample_data[key].to(self.device)
            with utils.TraceDict(
                module=self.model,
                layers=sample_layers,
                retain_input=False,
                retain_output=True,
            ) as tr:
                self(**sample_data)
            del sample_data
            mlp_logits = []
            adapter_logits = []
            for layer in self.hparams.layers:
                layer_adapter_logits, layer_mlp_logits = \
                    tr['roberta.encoder.layer.{}'.format(layer)].output
                layer_adapter_logits = layer_adapter_logits[0]
                mlp_logits.append(layer_mlp_logits)
                adapter_logits.append(layer_adapter_logits)
            mlp_logits = torch.stack(mlp_logits, dim=1)
            adapter_logits = torch.stack(adapter_logits, dim=1)
            batch_sample_loss = self.mse_loss(mlp_logits, adapter_logits)
            batch_sample_loss = batch_sample_loss.mean(
                dim=-1).mean(dim=-1).mean(dim=0).mean()
            sample_loss = batch_sample_loss
        else:
            sample_loss = 0.0
        
        # Compute knowledge loss
        knowledge_loss = self(**knowledge_data).loss
        
        # Compute total loss
        loss = self.hparams.knowledge_loss_factor * knowledge_loss + sample_loss
        
        self.log('knowledge_loss', knowledge_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('sample_loss', sample_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss}
        
    def on_train_start(self):
        if self.hparams.reweighting:
            self.weights = WeightTensor(len(self.knowledge_dataset), self.hparams.batch_size) \
                .to(self.device)
            self.weights.train(True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        return optimizer
        
    def on_train_epoch_end(self) -> None:
        self.model.save_knowledge_adapter(self.model_filename + str(self.current_epoch))
        
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
    