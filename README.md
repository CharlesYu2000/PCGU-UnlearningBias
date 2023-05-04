# Unlearning Bias in Language Models by Partitioning Gradients

This repo contains the code and experiments for _Partitioned Contrastive Gradient Unlearning (PCGU)_, a method proposed in the paper Unlearning Bias in Language Models by Partitioning Gradients (Findings of ACL 2023). 

## Environment
While running in the desired pip or conda environment, run `conda install --file requirements.txt` or `pip install -r requirements.txt`. 

## Downloading the Data
1. Download the WinoGender data from https://github.com/rudinger/winogender-schemas/blob/master/data/all_sentences.tsv and https://github.com/rudinger/winogender-schemas/blob/master/data/occupations-stats.tsv to `data/wg.tsv` and `data/wg_stats.tsv` respectively.
2. Download the CrowS data from https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv to `data/crows_pairs_anonymized.csv`.
3. Download the StereoSet dev data from https://github.com/moinnadeem/StereoSet/blob/master/data/dev.json to `data/stereoset_dev.json`. 

## Processing the StereoSet data
Here, we will prepare the StereoSet data for use in our evaluation/model selection scripts. 

In the `data` directory, run `python prepare_ss.py` to generate the files used for evaluation. If you want unshuffled data, in `prepare_ss.py`, change the `SHUFFLE` variable to be `False`. 

## Unlearning using PCGU
The main finetuning loop of PCGU can be found at `src/general_similarity_retrain.py`. This file is the place to look if you're interested in extending/modifying PCGU. In the paper, we used the simplest training procedure possible, meaning no learning rate scheduler, so the only important argument to that script is `-n`, which allows you to train for any number of epochs. Each epoch is cheap to train, so feel free to play around with this script to train many versions of a model and analyze the results (note that this can use a lot of storage, so remember to clear out checkpoints after use, as this script does not overwrite checkpoints for the same configuration of model). 

This script should train models and dump their checkpoints to `src/models`. 

## Evaluating PCGU-unlearned models
After training models with `src/general_similarity_retrain.py`, you can evaluate the trained models using the `src/evaluate_models.py` script. This script should automatically find the models of `src/models` and dump StereoSet evaluation results to `src/results`. It should also be quite simple to create ones own evaluation by using the `eval_mlm_given_dataloader()` and `compute_stereoset_scores()` functions of `src/eval_on_stereoset.py`. 

To evaluate a model on CrowS, use the script `src/eval_on_crows.py`. 

## Reference
If you used/built on top of/adapted PCGU in your work, we would appreciate if you cited our paper directly! 
```
@inproceedings{yu-2023-unlearning,
    title     = {Unlearning Bias in Language Models by Partitioning Gradients},
    author    = {Yu, Charles and Jeoung, Sullam and Kasi, Anish and Yu, Pengfei and Ji, Heng},
    year      = {2023},
    booktitle = {Proc. The 61st Annual Meeting of the Association for Computational Linguistics (ACL2023) Findings}
}
```