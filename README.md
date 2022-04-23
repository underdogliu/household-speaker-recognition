# subsets_adaptation


### TODO::CODE

| Issue | Comment | Status 
|-|-|-|
| Create train/dev/test_enroll/test_adapt/test_eval splits for the Voxceleb1 dataset | train is for thraining the scoring backend, dev can be used for tuning some hyperparams, calibration, etc, and test is for households simulation |  |
| Create evaluation protocols for the Voxceleb1 dataset |  |  |
| Create protocols for score calibration (for each dataset), save to .txt |  |  |
| Unified dataloading such that the main script is independent on the dataset (ASVSpoof, Voxceleb, etc) | See the best practices in opensource projects. Perhaps, implement a Dataset class with methods like `load(path, part="dev")` |  |
| Think about the unified interface for models,  preferably to support both active and passive enrollments | All model classes should have the same set of methods   like `fit()`, `predict()`, etc - something like in sklearn  |  |
| Better way to keep and present the metrics |  |  |
| Implement saving scores to the disk |  |  |
| Implement loading of model parameters (e.g. PLDA) in the main script | Maybe have a shared .yaml config with general settings and a separate .yaml config files for each algorithm? |  |
| Implement scoring for full PLDA |  |  |
| Train full PLDA on TDNN embeddings |  |  |
| Implement evaluation for passive enrollment scenario |  |  |
| Implement as many baselines as possible | For example, [1] has 4 baselines and 3 proposed algorithms. We may implemet some of them as well as use other algorithms as baselines |  |
| Implement as many new algorithms as possible |  |  |
|  |  |  |


### TODO::PAPER
| Issue | Comment | Status 
|-|-|-|
| [1] - very similar work. Think about how our work is different to highlight potential advantages of our approach. | 1) we use a more realistic setup where the adaptation set can include the guest speakers; 2) we proposed an algorithm with _online_ updates; 3) what else? |  |
| [1] uses Label Propagation clustering which, as it seems, requires the number of clusters to be known. In other words, the adaptation set cannot include utterances  from the guest speakers (less realistic). Confirm that this is the case. |  |  |
| Think about visual materials to be included | 1) Plot error vs threshold 2) Plot error vs size of the adaptation set (as in [1]) 3) Score distribution histograms to demonstrate multi-enroll verification 4) what else? |  |
|  |  |  |


### References:

1. [Graph-based Label Propagation for Semi-Supervised Speaker Identification](https://arxiv.org/pdf/2106.08207.pdf) 
2. [Improving Speaker Identification for Shared Devices
by Adapting Embeddings to Speaker Subsets](https://arxiv.org/pdf/2109.02576.pdf)


