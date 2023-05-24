from argparse import ArgumentParser
from datetime import datetime
from datasets import load_dataset
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup
import torch

from src.data.combined_dataset import CombinedDataset

from .transformers import RobertaForMaskedLM, RobertaConfig, WeightTensor
from ..data.domain_knowledge_dataset import DomainKnowledgeDataset
from ..data.tok_dataset import TokenizedDataset
from .. import utils
import numpy as np
from copy import deepcopy
from ..realm.realm import RealmRetriever

class MixDAStageOne(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--dirpath", type=str, default="models/mod_stage_one"
        )
        parser.add_argument(
            "--knowledge_data_path",
            type=str,
            default="datasets/dataset/train.json",
        )
        parser.add_argument("--batch_size", type=int, default=20)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--moe_lr", type=float, default=5e-4)
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
        parser.add_argument("--gating_module", type=str, default='roberta.encoder.layer.{}.output.gating')
        parser.add_argument("--mlp_module", type=str, default='roberta.encoder.layer.{}.output')
        parser.add_argument("--load_adapters", type=str)
        parser.add_argument("--adapter_down_scale", type=float, default=1.0)
        
        parser.add_argument("--realm_record", type=str)
        parser.add_argument("--no_new_knowledge", default=False, action='store_true')
        parser.add_argument("--disable_moe", default=False, action='store_true')
        parser.add_argument("--no_old_knowledge", default=False, action='store_true')

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
            disable_moe=self.hparams.disable_moe,
            num_of_kas=1,
        )
        self.model = RobertaForMaskedLM.from_pretrained(self.hparams.model_name, config=model_config).to(self.device)
        self.ka_list = None if self.hparams.load_adapters is None or self.hparams.load_adapters == '' else [
            x for x in self.hparams.load_adapters.split(',')]
        
        if self.ka_list is not None and len(self.ka_list) > 0:
            self.model.load_knowledge_adapter(self.ka_list)
            print('adapter loaded!!!')
        
        # prepare loss functions
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        
        self.model_filename = self.hparams.dirpath + '/model' + datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S') + '.pt'
        
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
            if not self.hparams.disable_moe:
                layer_list.append('model.' + self.hparams.gating_module.format(layer))
        # utils.enable_grad_by_module(self.model, layer_list)
        utils.enable_grad_by_module(self, layer_list)
        
        
    def train_dataloader(self, shuffle=False):
        if not hasattr(self, 'knowledge_dataset'):
            self.knowledge_dataset = DomainKnowledgeDataset(
                self.hparams.knowledge_data_path,
                self.hparams.dataset_size_limit
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
        self.train_dataset = CombinedDataset(self.knowledge_dataset, self.sample_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.train_dataset.collate_fn,
            shuffle=shuffle,
            num_workers=16,
            drop_last=True
        )
        
    def val_dataloader(self, shuffle=False):
        if not hasattr(self, 'knowledge_dataset_val'):
            self.knowledge_dataset_val = DomainKnowledgeDataset(
                self.hparams.knowledge_data_path,
                1000
            )
        return DataLoader(
            self.knowledge_dataset_val,
            batch_size=self.hparams.batch_size,
            collate_fn=self.knowledge_dataset_val.collate_fn,
            shuffle=shuffle,
            num_workers=16
        )
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    
    def training_step(self, batch, batch_idx=None):
        knowledge_data, sample_data = batch
        sample_layers = ['roberta.encoder.layer.{}'.format(layer) for layer in self.hparams.layers]
        # Compute knowledge loss
        if not self.hparams.no_new_knowledge:
            ans_inputs = [prompt.format(*targets).strip() for prompt, targets in zip(knowledge_data['prompt'], knowledge_data['targets'])]
            
            all_ans_tok = self.tokenizer(
                ans_inputs,
                return_tensors='pt',
                padding=True,
                return_offsets_mapping=True,
            ).to(self.device)
            offset_mapping = all_ans_tok.pop('offset_mapping')
            all_input_tok = deepcopy(all_ans_tok)
            for tok, offset, ans_input, targets in zip(all_input_tok['input_ids'], offset_mapping, ans_inputs, knowledge_data['targets']):
                for target in targets:
                    pos_begin = ans_input.find(target)
                    pos_end = pos_begin + len(target)
                    for i, r in enumerate(offset):
                        b, e = r
                        if b == e:
                            continue
                        if e < pos_begin:
                            continue
                        elif b > pos_end:
                            break
                        tok[i] = self.tokenizer.mask_token_id
            logits = self(**all_input_tok)[0]
            
            ans_input_ids = all_ans_tok['input_ids']
            all_input_logits = logits.view(-1, logits.shape[-1])
            all_ans_logits = ans_input_ids.view(-1)
            knowledge_loss_each = self.ce_loss(all_input_logits, all_ans_logits).view(ans_input_ids.shape)

            mask = (all_input_tok['input_ids'] == self.tokenizer.mask_token_id).long()
            knowledge_loss_each *= mask
            knowledge_loss = knowledge_loss_each.sum(dim=-1) / mask.sum(dim=-1)
            knowledge_loss = knowledge_loss.view(-1, 1)
            knowledge_loss = knowledge_loss.mean(dim=1)
            if self.hparams.reweighting:
                knowledge_loss = (knowledge_loss * self.weights(batch_idx)).sum()
            else:
                knowledge_loss = knowledge_loss.mean()
        else:
            knowledge_loss = 0.0
            
        # Compute sample loss
        if not self.hparams.no_old_knowledge:
            if self.realm is not None:
                # Fetch "sample_data" from REALM
                ans_rep = self.model(all_ans_tok['input_ids'], output_hidden_states=True).hidden_states[-1][:, 0]
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
            batch_sample_loss = batch_sample_loss.mean(dim=-1).mean(dim=-1).mean(dim=0).mean()
            sample_loss = batch_sample_loss
        else:
            sample_loss = 0.0
        
        # Compute total loss
        loss = self.hparams.knowledge_loss_factor * knowledge_loss + sample_loss
        
        self.log('knowledge_loss', knowledge_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('sample_loss', sample_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx=None):
        exact_match, match_score = [], []
        knowledge_data = batch
        # Input with Prompts + masks
        ans_inputs = [prompt.format(*targets).strip() for prompt, targets in zip(knowledge_data['prompt'], knowledge_data['targets'])]
        
        all_ans_tok = self.tokenizer(
            ans_inputs,
            return_tensors='pt',
            padding=True,
            return_offsets_mapping=True,
        ).to(self.device)
        offset_mapping = all_ans_tok.pop('offset_mapping')
        all_input_tok = deepcopy(all_ans_tok)
        for tok, offset, ans_input, targets in zip(all_input_tok['input_ids'], offset_mapping, ans_inputs, knowledge_data['targets']):
            for target in targets:
                pos_begin = ans_input.find(target)
                pos_end = pos_begin + len(target)
                for i, r in enumerate(offset):
                    b, e = r
                    if b == e:
                        continue
                    if e < pos_begin:
                        continue
                    elif b > pos_end:
                        break
                    tok[i] = self.tokenizer.mask_token_id                    
        
        all_results = self(**all_input_tok)
        logits = all_results[0]
        ans_input_ids = all_ans_tok['input_ids']
        all_input_logits = logits
        
        mask = (all_input_tok['input_ids'] == self.tokenizer.mask_token_id).long()
        all_input_softmax = all_input_logits.log_softmax(dim=-1)
        all_input_ans = torch.argmax(all_input_softmax, dim=-1) * mask
        all_correct_ans = ans_input_ids * mask
        for input_ans, correct_ans, input_softmax in zip(all_input_ans, all_correct_ans, all_input_softmax):
            if torch.equal(input_ans, correct_ans):
                exact_match.append(1)
            else:
                exact_match.append(0)
            correct_ans_nonzero_pos = torch.argwhere(correct_ans).T
            softmax_scores = input_softmax[correct_ans_nonzero_pos[0], correct_ans[correct_ans.nonzero()].flatten()]
            softmax_score = softmax_scores.mean(dim=-1)
            match_score.append(round(torch.exp(softmax_score).item() * 100, 2))
            
        mean_exact_match = round(np.mean(exact_match) * 100, 2)
        mean_match_score = round(np.mean(match_score), 2)
        
        self.log('exact_match', mean_exact_match, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch['id']))
        self.log('match_score', mean_match_score, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(batch['id']))
        
    def on_train_start(self):
        if self.hparams.reweighting:
            self.weights = WeightTensor(len(self.knowledge_dataset), self.hparams.batch_size) \
                .to(self.device)
            self.weights.train(True)
    
    def configure_optimizers(self):
        param_list = []
        gate_param_list = []
        for layer in self.hparams.layers:
            adapter = utils.get_module(self.model, self.hparams.adapter_module.format(layer))
            if not self.hparams.disable_moe:
               gate = utils.get_module(self.model, self.hparams.gating_module.format(layer))
            for p in adapter.parameters():
                param_list.append(p)
            if not self.hparams.disable_moe:
                for p in gate.parameters():
                    gate_param_list.append(p)
        if self.realm is not None:
            for p in self.realm.parameters():
                param_list.append(p)
        optimizer = torch.optim.AdamW([
            {
                'params': param_list,
                'lr': self.hparams.lr,
            },
            {
                'params': gate_param_list,
                'lr': self.hparams.moe_lr,
            }
        ], weight_decay= self.hparams.weight_decay)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_updates,
        #     num_training_steps=self.hparams.total_num_updates,
        # )

        return optimizer
        
    def on_train_epoch_end(self) -> None:
        self.model.save_knowledge_adapter(self.model_filename + '-' + str(self.current_epoch))

