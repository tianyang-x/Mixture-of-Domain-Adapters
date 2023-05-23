# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🥹 Transformers model for sequence classification on GLUE. 🥹🥹🥹"""
import argparse
from ast import Pass
import json
import logging
import math
import os
import random
from pathlib import Path
from xml.sax.handler import all_properties

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HoulsbyConfig,
    PfeifferConfig,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    PrefixTuningConfig,
)
import numpy as np
from transformers.utils.versions import require_version
import wandb
from datasets import Dataset

import sys
sys.path.append('/home/nlpintern/xty/KnowledgeEditor')
from src import utils

from src.models.transformers import RobertaForSequenceClassification, RobertaConfig
from src.realm.realm import RealmRetriever
from opendelta import LoraModel

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default='qnli',
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default='roberta-large'
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🥹 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--load_romes",
        type=str,
        help="Load ROME knowledge adapter."
    )
    parser.add_argument('--load_task_adapter', type=str)
    parser.add_argument(
        "--layers",
        type=str,
        help="Layers modified by ROME."
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        choices=['pfeiffer', 'houlsby', 'lora', 'prefix_tuning'],
        default='houlsby',
        help="Type of adapter for downstream tasks"
    )
    parser.add_argument(
        "--adapter_non_linearity",
        type=str,
        choices=['relu', 'none', 'swish'],
        help="non-linear activation function inside the adapter."
    )
    parser.add_argument(
        '--reduction_factor',
        type=int,
        help='Reduction factor in MLP inside the adapter.'
    )
    parser.add_argument(
        '--lora_r',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--project_name',
        type=str,
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--realm_record',
        type=str
    )
    parser.add_argument(
        '--disable_moe',
        action='store_true'
    )
    parser.add_argument(
        '--few_shot',
        type=int,
        default=16
    )
    args = parser.parse_args()
    if args.layers is None:
        args.layers = []
    else:
        args.layers = [int(item) for item in args.layers.split(',')]

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    wandb.login(key='710f9ed51f388218c59dda998f08db93f481da29')
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    romes = args.load_romes.split(',') if args.load_romes is not None else []
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name, layers=args.layers, enable_attention=True, enable_old_ka=True, num_of_kas=len(romes), disable_moe=args.disable_moe,)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
    )
    
    # Add adapter and load knowledge adapter
    if args.load_romes != '' and args.load_romes is not None:
        romes = args.load_romes.split(',')
        model.load_knowledge_adapter(romes)
        print('ROME loaded!!1')
    else:
        print('Disabling parallel knowledge adapter')
    if args.load_task_adapter is not None:
        model.load_adapter(args.load_task_adapter)
        model.train_adapter('domain', 'pfeiffer')
    elif args.adapter_type == 'pfeiffer':
        adapter_config = PfeifferConfig(
            ln_after=False,
            ln_before=False,
            mh_adapter=False,
            output_adapter=True,
            adapter_residual_before_ln=False,
            non_linearity=args.adapter_non_linearity,
            original_ln_after=True,
            original_ln_before=True,
            reduction_factor=16,
            residual_before_ln=True
        )
    elif args.adapter_type == 'houlsby':
        adapter_config = HoulsbyConfig(
            ln_after=False,
            ln_before=False,
            mh_adapter=True,
            output_adapter=True,
            adapter_residual_before_ln=False,
            non_linearity=args.adapter_non_linearity,
            original_ln_after=True,
            original_ln_before=False,
            reduction_factor=16,
            residual_before_ln=True
        )
    elif args.adapter_type == 'lora':
        delta_model = LoraModel(model, args.lora_r, args.lora_alpha)
        delta_model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)
    elif args.adapter_type == 'prefix_tuning':
            adapter_config = PrefixTuningConfig(flat=False, prefix_length=30)
    else:
        raise NotImplementedError()
    if args.adapter_type != 'lora' and args.load_task_adapter is None:
        model.add_adapter(args.task_name, args.adapter_type, adapter_config)
        model.train_adapter(args.task_name, args.adapter_type)
    # We train MOE gate parameters!
    if not args.disable_moe:
        for layer in args.layers:
            moe_gate = model.roberta.encoder.layer[layer].output.gating
            for param in moe_gate.parameters():
                param.requires_grad_(True)
            
    # enable REALM
    realm = None
    if args.realm_record is not None:
        doc_records = np.load(args.realm_record)
        realm = RealmRetriever(
            doc_records,
            model_hidden_size=model.config.hidden_size,
            query_embed_size=128,
            doc_proj_size=128,
            num_docs=len(doc_records)
        ).to(model.device)
        sample_dataset = load_dataset(
            'wikipedia', '20220301.en'
        )['train']
    
    print('=' * 50)
    print('The following parameters are not frozen:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print('=' * 50)

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        result = {}
        if sentence2_key is not None:
            result['text2'] = examples[sentence2_key]
        result['text1'] = examples[sentence1_key]

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    
    def preprocess_function_k_shot(examples, few_shot):
        random_indices = list(range(0, len(examples["labels"])))
        random.shuffle(random_indices)

        new_examples = {}
        for key in examples.features.keys():
            new_examples[key] = []
        label_count = {}
        
        not_hit = 0

        for index in random_indices:      
            label = examples['labels'][index]
            if label not in label_count:
                label_count[label] = 0
                not_hit = 0
            else:
                not_hit += 1
                if not_hit == 1000:
                    break

            if label_count[label] < few_shot:
                for key in examples.features.keys():
                    new_examples[key].append(examples[key][index])
                label_count[label] += 1
        
        print('k-shot selection done!!')
        
        return Dataset.from_dict(new_examples)
    
    if args.few_shot is not None:
        train_dataset = preprocess_function_k_shot(train_dataset, args.few_shot)
        
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else []))
        pass

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    moe_weights = []
    if not args.disable_moe:
        for layer in args.layers:
            for p in model.roberta.encoder.layer[layer].output.gating.parameters():
                moe_weights.append(p)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if realm is not None:
        optimizer_grouped_parameters.append({
            "params": realm.parameters(),
            "weight_decay": args.weight_decay,
        })
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model,  optimizer, train_dataloader, eval_dataloader, lr_scheduler, realm = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, realm
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if args.with_tracking:
        if accelerator.is_main_process:
            experiment_config = vars(args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            accelerator.init_trackers(args.project_name, experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name)
    else:
        metric = load_metric("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # record metrics
    metrics = ['f1', 'accuracy'] if args.task_name != 'stsb' else ['pearson', 'spearmanr']
    best_metrics = { k: 0.0 for k in metrics }
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            if sentence2_key is None:
                tok = tokenizer(batch['text1'], padding=True, truncation=True, return_tensors='pt').to(model.device)
            else:
                tok = tokenizer(batch['text1'], batch['text2'], padding=True, truncation=True, return_tensors='pt').to(model.device)
            if realm is not None:
                text_rep = model(tok['input_ids'], output_hidden_states=True).hidden_states[-1][:, 0]
                ids = realm(text_rep, 1)
                ids = ids.flatten().cpu().tolist()
                items = [sample_dataset[id] for id in ids]
                for i in range(len(batch['text1'])):
                    all_texts = []
                    for j in range(1):
                        index = i * 1 + j
                        doc = items[index]['text'].split('.')    
                        all_texts.append(' '.join(doc[:3]))
                    all_texts_str = '</s>'.join(all_texts)
                    batch['text1'][i] = all_texts_str + '</s>' + batch['text1'][i]
            if sentence2_key is None:
                tok = tokenizer(batch['text1'], padding=True, truncation=True, return_tensors='pt').to(model.device)
            else:
                tok = tokenizer(batch['text1'], batch['text2'], padding=True, truncation=True, return_tensors='pt').to(model.device)
            if args.task_name == 'stsb':
                tok['labels'] = batch['labels'].float()
            else:
                tok['labels'] = batch['labels']
            outputs = model(**tok)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        moe_layers = ['roberta.encoder.layer.{}.output.gating'.format(layer) for layer in args.layers] if not args.disable_moe else []
        all_rome_weights = []
        for step, batch in enumerate(eval_dataloader):
            if sentence2_key is None:
                tok = tokenizer(
                    batch['text1'], padding=True, truncation=True, return_tensors='pt').to(model.device)
            else:
                tok = tokenizer(batch['text1'], batch['text2'], padding=True,
                                truncation=True, return_tensors='pt').to(model.device)
            if realm is not None:
                text_rep = model(
                    tok['input_ids'], output_hidden_states=True).hidden_states[-1][:, 0]
                ids = realm(text_rep, 1)
                ids = ids.flatten().cpu().tolist()
                items = [sample_dataset[id] for id in ids]
                for i in range(len(batch['text1'])):
                    all_texts = []
                    for j in range(1):
                        index = i * 1 + j
                        doc = items[index]['text'].split('.')
                        all_texts.append(' '.join(doc[:3]))
                    all_texts_str = '</s>'.join(all_texts)
                    batch['text1'][i] = all_texts_str + \
                        '</s>' + batch['text1'][i]
            if sentence2_key is None:
                tok = tokenizer(
                    batch['text1'], padding=True, truncation=True, return_tensors='pt').to(model.device)
            else:
                tok = tokenizer(batch['text1'], batch['text2'], padding=True,
                                truncation=True, return_tensors='pt').to(model.device)
            if args.task_name == 'stsb':
                tok['labels'] = batch['labels'].float()
            else:
                tok['labels'] = batch['labels']
            with torch.no_grad():
                with utils.TraceDict(
                    module=model.module if hasattr(model, 'module') else model,
                    layers=moe_layers,
                    retain_input=False,
                    retain_output=True,
                ) as tr:
                    outputs = model(**tok)
            if not args.disable_moe:
                for layer in args.layers:
                    weights = tr['roberta.encoder.layer.{}.output.gating'.format(layer)].output[:,:,0]
                    for w in weights:
                        all_rome_weights.append(w.item())
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}: {eval_metric}, rome_weights: {np.mean(all_rome_weights)} +- {np.std(all_rome_weights)}")
        
        # for m in metrics:
        #     best_metrics[m] = max(best_metrics[m], eval_metric[m])

        if args.with_tracking:
            accelerator.log(
                {
                    "accuracy" if args.task_name is not None else "glue": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                    'rome_weights_mean': np.mean(all_rome_weights),
                    'rome_weights_std': np.std(all_rome_weights),
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")

    # for m in metrics:
    #     print("#####" + m + '####:' + str(best_metrics[m]))


if __name__ == "__main__":
    main()
