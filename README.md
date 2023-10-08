# Mixture-of-Domain-Adapters

Code for *Mixture-of-Domain-Adapters: Decoupling and Injecting Domain Knowledge to Pre-trained Language Models' Memories* (ACL 2023, in camera ready).

## Incompatibilities with New Versions of PyTorch Lightning
PyTorch Lightning made several breaking changes incompatible to the existing code in ver. 2.0.0. For now, please run the code with `pytorch_lightning==1.9.0`.

## Datasets
Datasets can be found [here](https://purdue0-my.sharepoint.com/:f:/g/personal/xu1868_purdue_edu/Ethu6AEk5V1IgRVh5BSLj64BncZl5WRPIgxKAHpLgxltlw). Pretrained Stage 1 models are [here](https://purdue0-my.sharepoint.com/:f:/g/personal/xu1868_purdue_edu/EitjSv7bE2RAv6VTI9EhBsABZiugl87UpQ8HMHWtRk6PGg).

For Stage 1, datasets are in json files like:
```json
[
  {
    "prompt": "This is a prompt with targets like this to be {} .",
    "targets": [
      "filled"
    ]
  },
]
```

For Stage 2 (classification tasks), datasets are in jsonl files like:
```json
{"text": "This is a helpful sentence.", "label": "helpful"}
```

You can modify the code to accommodate the model to your dataset.

## Running the Code
Please refer to instructions in `stage_one_pretrain.sh` and `stage_two.sh`, which give examples on how to execute Stage 1 and Stage 2 training respectively.

## Citation
If you use or extend our work, please cite the following [paper](https://arxiv.org/abs/2306.05406):
```
@article{diao2023mixture,
  title={Mixture-of-Domain-Adapters: Decoupling and Injecting Domain Knowledge to Pre-trained Language Models Memories},
  author={Diao, Shizhe and Xu, Tianyang and Xu, Ruijia and Wang, Jiawei and Zhang, Tong},
  journal={arXiv preprint arXiv:2306.05406},
  year={2023}
}
```

## Questions?
Please raise your questions in the issue or direct them to `xu1868@purdue.edu`.