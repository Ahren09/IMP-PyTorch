# Infinite Mixture Prototypes

## Updates

- 2023.8.14: Updated code

## Installation

- Install PyTorch
- `conda install -c conda-forge imageio h5py`



Available config registries

```
{'mini-imagenet_basic': fewshot.configs.mini_imagenet_config.BasicConfig,
 'mini-imagenet_kmeans-refine': fewshot.configs.mini_imagenet_config.KMeansRefineConfig,
 'mini-imagenet_kmeans-distractor': fewshot.configs.mini_imagenet_config.KMeansDistractorConfig,
 'mini-imagenet_imp': fewshot.configs.mini_imagenet_config.ImpModelConfig,
 'mini-imagenet_crp': fewshot.configs.mini_imagenet_config.CRPConfig,
 'mini-imagenet_map-dp': fewshot.configs.mini_imagenet_config.MapDPConfig,
 'mini-imagenet_soft-nn': fewshot.configs.mini_imagenet_config.SoftNNConfig,
 'mini-imagenet_dp-means-hard': fewshot.configs.mini_imagenet_config.DPMeansHardConfig,
 
 'omniglot_basic': fewshot.configs.omniglot_config.BasicConfig,
 'omniglot_kmeans-refine': fewshot.configs.omniglot_config.KMeansRefineConfig,
 'omniglot_kmeans-distractor': fewshot.configs.omniglot_config.KMeansDistractorConfig,
 'omniglot_imp': fewshot.configs.omniglot_config.ImpModelConfig,
 'omniglot_crp': fewshot.configs.omniglot_config.CRPConfig,
 'omniglot_map-dp': fewshot.configs.omniglot_config.MapDPConfig,
 'omniglot_soft-nn': fewshot.configs.omniglot_config.SoftNNConfig,
 'omniglot_dp-means-hard': fewshot.configs.omniglot_config.DPMeansHardConfig,
 
 'tiered-imagenet_basic': fewshot.configs.tiered_imagenet_config.BasicConfig,
 'tiered-imagenet_kmeans-refine': fewshot.configs.tiered_imagenet_config.KMeansRefineConfig,
 'tiered-imagenet_kmeans-distractor': fewshot.configs.tiered_imagenet_config.KMeansDistractorConfig,
 'tiered-imagenet_imp': fewshot.configs.tiered_imagenet_config.ImpModelConfig,
 'tiered-imagenet_crp': fewshot.configs.tiered_imagenet_config.CRPConfig,
 'tiered-imagenet_map-dp': fewshot.configs.tiered_imagenet_config.MapDPConfig,
 'tiered-imagenet_soft-nn': fewshot.configs.tiered_imagenet_config.SoftNNConfig,
 'tiered-imagenet_dp-means-hard': fewshot.configs.tiered_imagenet_config.DPMeansHardConfig}
```

## Code
This repository is adapted from https://github.com/renmengye/few-shot-ssl-public for PyTorch 0.3.1


### Usage Examples
`submit_omniglot.sh` provides example usage of the main file.

We also have submission scripts for running code on a slurm cluster. 
Please refer to `submit_all_models.sh` and `submit_super.sh` for examples.
