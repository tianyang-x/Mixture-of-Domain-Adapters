from argparse import ArgumentParser
from cProfile import label

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
from src.data.task_dataset import TaskDataset

from src.models.transformers.modeling_roberta import RobertaForSequenceClassification
from .transformers import RobertaForSequenceClassification, RobertaConfig
from torchmetrics import Accuracy, F1Score
from transformers.adapters.configuration import HoulsbyConfig, PfeifferConfig
from .. import utils
from transformers.adapters import PrefixTuningConfig
import numpy as np
from opendelta import LoraModel
from datetime import datetime
from ..realm.realm import RealmRetriever
from ..data.tok_dataset import TokenizedDataset
from datasets import load_dataset


class MoDClassification(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_data_path",
            default="datasets/dataset/train.jsonl",
            type=str,
        )
        parser.add_argument(
            "--dev_data_path",
            default="datasets/dataset/dev.jsonl",
            type=str,
        )
        parser.add_argument(
            "--test_data_path",
            default="datasets/dataset/test.jsonl",
            type=str,
        )
        parser.add_argument("--batch_size", type=int, default=3)
        parser.add_argument("--lr", type=float, default=3e-5)
        parser.add_argument("--weight_decay", type=int, default=0.01)
        parser.add_argument("--total_num_updates", type=int, default=10000)
        parser.add_argument("--warmup_updates", type=int, default=500)
        parser.add_argument("--num_workers", type=int, default=32)

        parser.add_argument("--model_name", type=str,
                            default="roberta-large")

        parser.add_argument('--load_adapters', type=str)
        parser.add_argument('--adapter_type', type=str)
        parser.add_argument('--adapter_non_linearity', type=str, default='swish')
        parser.add_argument('--load_task_adapter', type=str)
        parser.add_argument('--layers', type=str)
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=8)
        parser.add_argument(
            "--dirpath", type=str, required=True
        )
        parser.add_argument("--realm_record", type=str)
        parser.add_argument('--realm_top_k', type=int, default=1)
        parser.add_argument("--disable_moe", default=False, action='store_true')
        parser.add_argument('--few_shot', type=int)
        return parser

    def __init__(self, config=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.layers = [] if self.hparams.layers is None or self.hparams.layers == '' else [int(x) for x in self.hparams.layers.split(',')]
        
        self.train_dataset = TaskDataset(data_path=self.hparams.train_data_path, few_shot=self.hparams.few_shot)
        self.val_dataset = TaskDataset(data_path=self.hparams.dev_data_path, label2id=self.train_dataset.label2id, few_shot=self.hparams.few_shot)
        if self.hparams.test_data_path is not None:
            self.test_dataset = TaskDataset(data_path=self.hparams.test_data_path, label2id=self.train_dataset.label2id)
        else:
            self.test_dataset = None
        
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.hparams.model_name)
        self.ka_list = [] if self.hparams.load_adapters is None or self.hparams.load_adapters == '' else [x for x in self.hparams.load_adapters.split(',')]
        model_config = RobertaConfig.from_pretrained(
            self.hparams.model_name,
            layers=self.hparams.layers,
            output_original=True,
            num_labels=len(self.train_dataset.label2id),
            enable_old_ka=True,
            num_of_kas=len(self.ka_list) if self.ka_list is not None else 0,
            disable_moe=self.hparams.disable_moe,
        )
        self.model = RobertaForSequenceClassification.from_pretrained(self.hparams.model_name, config=model_config).to(self.device)
        if self.ka_list is not None and len(self.ka_list) > 0:
            self.model.load_knowledge_adapter(self.ka_list)
            print('adapter loaded!!!')

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_f1 = F1Score(average='macro', num_classes=len(self.train_dataset.label2id))
        self.valid_f1 = F1Score(average='macro', num_classes=len(self.train_dataset.label2id))
        self.test_f1 = F1Score(average='macro', num_classes=len(self.train_dataset.label2id))
        self.ce = torch.nn.CrossEntropyLoss()
        
        if self.hparams.load_task_adapter is not None:
            self.model.load_adapter(self.hparams.load_task_adapter)
            self.model.train_adapter('domain', 'pfeiffer')
        elif self.hparams.adapter_type == 'pfeiffer':
            adapter_config = PfeifferConfig(
                ln_after=False,
                ln_before=False,
                mh_adapter=False,
                output_adapter=True,
                adapter_residual_before_ln=False,
                non_linearity=self.hparams.adapter_non_linearity,
                original_ln_after=True,
                original_ln_before=True,
                reduction_factor=16,
                residual_before_ln=True
            )
        elif self.hparams.adapter_type == 'houlsby':
            adapter_config = HoulsbyConfig(
                ln_after=False,
                ln_before=False,
                mh_adapter=True,
                output_adapter=True,
                adapter_residual_before_ln=False,
                non_linearity=self.hparams.adapter_non_linearity,
                original_ln_after=True,
                original_ln_before=False,
                reduction_factor=16,
                residual_before_ln=True
            )
        elif self.hparams.adapter_type == 'lora':
            self.delta_model = LoraModel(self.model, self.hparams.lora_r, self.hparams.lora_alpha)
            self.delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
        elif self.hparams.adapter_type == 'prefix_tuning':
            adapter_config = PrefixTuningConfig(flat=False, prefix_length=30)
        if self.hparams.adapter_type != 'lora' and self.hparams.adapter_type is not None and self.hparams.load_task_adapter is None:
            self.model.add_adapter('domain', self.hparams.adapter_type, adapter_config)
            self.model.train_adapter('domain', self.hparams.adapter_type)
            
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
            
        # We train MOE gate parameters!
        if not self.hparams.disable_moe:
            for layer in self.hparams.layers:
                moe_gate = self.model.roberta.encoder.layer[layer].output.gating
                for param in moe_gate.parameters():
                    param.requires_grad_(True)
        
        print('=' * 50)
        print('The following parameters are not frozen:')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        print('=' * 50)
        
        self.sample_dataset = load_dataset(
            'wikipedia', '20220301.en'
        )['train']
        
        self.model_filename = self.hparams.dirpath + '/moe' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + '.pt'
        self.adapter_filename = self.hparams.dirpath + '/adapter'
        self.best_f1 = None
        
        self.best_valid_metrics = {'f1': 0.0, 'acc': 0.0}
        self.best_test_metrics = {'f1': 0.0, 'acc': 0.0}
        self.best_valid_test_metrics = {'f1': 0.0, 'acc': 0.0}

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            # num_workers=self.hparams.num_workers,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            # num_workers=self.hparams.num_workers,
        )
        
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.test_dataset.collate_fn,
                # num_workers=self.hparams.num_workers,
            )
        return None

    def forward(self, batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx=None):
        text, label = batch['text'], torch.LongTensor(batch['label']).to(self.device)
        tok = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # if REALM is there, fetch samples and patch them to the beginning
        if self.realm is not None:
            text_rep = self.model(tok['input_ids'], output_hidden_states=True).hidden_states[-1][:, 0]
            ids = self.realm(text_rep, self.hparams.realm_top_k)
            ids = ids.flatten().cpu().tolist()
            items = [self.sample_dataset[id] for id in ids]
            for i in range(len(batch['text'])):
                all_texts = []
                for j in range(self.hparams.realm_top_k):
                    index = i * self.hparams.realm_top_k + j
                    doc = items[index]['text'].split('.')    
                    all_texts.append(' '.join(doc[:3]))
                all_texts_str = '</s>'.join(all_texts)
                batch['text'][i] = all_texts_str + '</s>' + batch['text'][i]
        
        all_rome_weights = []
        moe_layers = ['roberta.encoder.layer.{}.output.gating'.format(layer) for layer in self.hparams.layers] if not self.hparams.disable_moe else []
        with utils.TraceDict(
            module=self.model,
            layers=moe_layers,
            retain_input=False,
            retain_output=True,
        ) as tr:
            logits = self.model(**tok).logits
        if not self.hparams.disable_moe:
            for layer in self.hparams.layers:
                weights = tr['roberta.encoder.layer.{}.output.gating'.format(layer)].output[:,:,0]
                for w in weights:
                    all_rome_weights.append(w.item())

        cr = self.ce(logits, label)

        loss = cr

        preds = logits.argmax(-1)
        self.log("loss", cr, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.hparams.batch_size)
        self.train_acc(preds, label)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.hparams.batch_size
        )
        self.train_f1(preds, label)
        self.log(
            "train_f1", self.train_f1, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.hparams.batch_size
        )
        self.log(
            "rome_weights_mean", np.mean(all_rome_weights), on_step=True, on_epoch=False, prog_bar=True, batch_size=self.hparams.batch_size
        )
        self.log(
            "rome_weights_std", np.std(all_rome_weights), on_step=True, on_epoch=False, prog_bar=True, batch_size=self.hparams.batch_size
        )

        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx=None):
        text, label = batch['text'], torch.LongTensor(batch['label']).to(self.device)
        tok = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        
        # if REALM is there, fetch samples and patch them to the beginning
        if self.realm is not None:
            text_rep = self.model(tok['input_ids'], output_hidden_states=True).hidden_states[-1][:, 0]
            ids = self.realm(text_rep, self.hparams.realm_top_k)
            ids = ids.flatten().cpu().tolist()
            items = [self.sample_dataset[id] for id in ids]
            for i in range(len(batch['text'])):
                all_texts = []
                for j in range(self.hparams.realm_top_k):
                    index = i * self.hparams.realm_top_k + j
                    doc = items[index]['text'].split('.')
                    all_texts.append(' '.join(doc[:3]))
                all_texts_str = '</s>'.join(all_texts)
                batch['text'][i] = all_texts_str + '</s>' + batch['text'][i]
        
        all_rome_weights = []
        moe_layers = ['roberta.encoder.layer.{}.output.gating'.format(layer) for layer in self.hparams.layers] if not self.hparams.disable_moe else []
        with utils.TraceDict(
            module=self.model,
            layers=moe_layers,
            retain_input=False,
            retain_output=True,
        ) as tr:
            logits = self.model(**tok).logits
        if not self.hparams.disable_moe:
            for layer in self.hparams.layers:
                weights = tr['roberta.encoder.layer.{}.output.gating'.format(layer)].output[:,:,0]
                for w in weights:
                    all_rome_weights.append(w.item())

        self.valid_acc.update(logits.sigmoid(), label)
        self.log(
            "valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
        )
        self.valid_f1.update(logits, label)
        self.log(
            "valid_f1", self.valid_f1, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
        )
        self.log(
            "val_rome_weights_mean", np.mean(all_rome_weights), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
        )
        self.log(
            "val_rome_weights_std", np.std(all_rome_weights), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
        )

        return {"logits": logits}

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.realm is not None:
            optimizer_grouped_parameters.append({
                'params': self.realm.parameters(),
                'weight_decay': self.hparams.weight_decay,
            })

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_updates,
            num_training_steps=self.hparams.total_num_updates,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]
        
    def on_validation_epoch_end(self):
        valid_f1, valid_acc = self.valid_f1.compute(), self.valid_acc.compute()
            
        if self.test_dataset is not None:
            all_rome_weights = []
            for batch in self.test_dataloader():
                text, label = batch['text'], torch.LongTensor(batch['label']).to(self.device)
                tok = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
                # if REALM is there, fetch samples and patch them to the beginning
                if self.realm is not None:
                    text_rep = self.model(tok['input_ids'], output_hidden_states=True).hidden_states[-1][:, 0]
                    ids = self.realm(text_rep, self.hparams.realm_top_k)
                    ids = ids.flatten().cpu().tolist()
                    items = [self.sample_dataset[id] for id in ids]
                    for i in range(len(batch['text'])):
                        all_texts = []
                        for j in range(self.hparams.realm_top_k):
                            index = i * self.hparams.realm_top_k + j
                            doc = items[index]['text'].split('.')    
                        all_texts.append(' '.join(doc[:3]))
                        all_texts_str = '</s>'.join(all_texts)
                        batch['text'][i] = all_texts_str + '</s>' + batch['text'][i]
                moe_layers = ['roberta.encoder.layer.{}.output.gating'.format(layer) for layer in self.hparams.layers] if not self.hparams.disable_moe else []
                with utils.TraceDict(
                    module=self.model,
                    layers=moe_layers,
                    retain_input=False,
                    retain_output=True,
                ) as tr:
                    logits = self.model(**tok).logits
                if not self.hparams.disable_moe:
                    for layer in self.hparams.layers:
                        weights = tr['roberta.encoder.layer.{}.output.gating'.format(layer)].output[:,:,0]
                        for w in weights:
                            all_rome_weights.append(w.item())

                self.test_acc(logits.sigmoid(), label)
                self.log(
                    "test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
                )
                self.test_f1(logits.sigmoid(), label)
                self.log(
                    "test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
                )
            self.log(
                "test_rome_weights_mean", np.mean(all_rome_weights), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
            )
            self.log(
                "test_rome_weights_std", np.std(all_rome_weights), on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size
        )
            
        if self.test_acc < self.best_test_metrics['acc']:
            self.best_test_metrics['acc'] = self.test_acc
        if self.test_f1 < self.best_test_metrics['f1']:
            self.best_test_metrics['f1'] = self.test_f1
        if valid_acc < self.best_valid_metrics['acc']:
            self.best_valid_metrics['acc'] = valid_acc
            self.best_valid_test_metrics['acc'] = self.test_acc
        if valid_f1 < self.best_valid_metrics['f1']:
            self.best_valid_metrics['f1'] = valid_f1
            self.best_valid_test_metrics['f1'] = self.test_f1
            
    def on_fit_end(self):
        print("#####" + 'valid_acc' + '####:' + str(self.best_valid_metrics['acc']))
        print("#####" + 'valid_f1' + '####:' + str(self.best_valid_metrics['f1']))
        print("#####" + 'test_acc' + '####:' + str(self.best_test_metrics['acc']))
        print("#####" + 'test_f1' + '####:' + str(self.best_test_metrics['f1']))
        print("#####" + 'valid_test_acc' + '####:' + str(self.best_valid_test_metrics['acc']))
        print("#####" + 'valid_test_f1' + '####:' + str(self.best_valid_test_metrics['f1']))
