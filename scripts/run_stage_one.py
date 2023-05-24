import os
import sys
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

import sys
import cProfile

from src.models.mixda_stage_one_mlm import MixDAStageOneMLM
from src.models.mixda_classification import MixDAClassification
from src.models.mixda_stage_one_wo_ka import MixDAAdapterAblation
import wandb

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--project_name", type=str, default="add-fever-complete")
    parser.add_argument("--run_name", type=str, default="test")

    parser = MixDAStageOneMLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args, _ = parser.parse_known_args()

    if args.seed is not None:
        seed_everything(seed=args.seed)

    wandb.login(key='710f9ed51f388218c59dda998f08db93f481da29')
    logger = WandbLogger(project=args.project_name, name=args.run_name, entity='amano-aki')

    callbacks = [
        LearningRateMonitor(
            logging_interval="step",
        ),
    ]

    trainer = Trainer.from_argparse_args(
        args, 
        logger=logger, 
        callbacks=callbacks,
        strategy = DDPStrategy,
        num_sanity_val_steps=0
    )

    model = MixDAStageOneMLM(**vars(args))

    trainer.fit(model)
    
