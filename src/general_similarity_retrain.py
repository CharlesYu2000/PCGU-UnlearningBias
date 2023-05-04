from sqlite3 import NotSupportedError
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForMaskedLM
import numpy as np
import random
import argparse
import logging
import os
from tqdm import tqdm
import json
import sys
from dataset import WGDataset
from utils import set_random_seed
from model_utils import get_params_map, get_all_model_grads, accumulate_grad
from consts import PAD_TOKEN, MASK_TOKEN
from partition_params import create_param_partition

logger = logging.getLogger(__name__)

# NOTE: this is the most expensive operation here
def compute_similarities(param_partition, grads_1, grads_2): 
    '''

    :param param_partition: list of (param_name, index) pairs denoting the partition of the weights
    :param grads_1: dict from param_name to gradient tensor
    :param grads_2: dict from param_name to gradient tensor
    '''

    sims = [] # same indexing as param_partition
    for param_name, indices in param_partition: 
        grad_1 = grads_1[param_name]
        grad_2 = grads_2[param_name]

        if grad_1 is None or grad_2 is None: 
            sims.append(torch.Tensor([5]).squeeze()) # so that it should not be picked as a param to keep
            continue

        # index into the grads if this param is partitioned
        if indices is not None: 
            grad_1 = grad_1[indices]
            grad_2 = grad_2[indices]

        cosine_sim = F.cosine_similarity(grad_1, grad_2, dim=-1).detach().cpu()
        sims.append(cosine_sim)

    return sims

def find_parameters_to_keep(param_partition, similarities, k=10000, return_target_indices=False):
    # k is the number of the lowest similarities

    sim_stack = torch.stack(similarities)

    # k = k or sim_stack.shape[-1]
    top_k_result = sim_stack.topk(k, largest=False, sorted=True)

    target_indices = [ind.item() for ind in top_k_result[1]]
    params_to_keep = [param_partition[ind] for ind in target_indices]

    if return_target_indices: 
        return params_to_keep, target_indices
    else: 
        return params_to_keep

def _minimize_grads_2(grads_1, grads_2): 
    return grads_2

def _maximize_grads_1(grads_1, grads_2): 
    return -grads_1

def _maximize_grads(grads_1, grads_2): 
    return -(grads_1+grads_2)

def update_model_param_grads(optimizer, model_params_map, params_to_keep, grads_1, grads_2, device, new_grad_calc=_minimize_grads_2): 
    '''
    By convention, grads_1 should be the "pos grads" and grads_2 should be the "neg grads"
    For new_grad_calc, remember that optimizer steps in the direction of minimization (i.e., opposite direction of what new_grad_calc returns)
    '''

    optimizer.zero_grad() # so that any grad not for param in params_to_keep is zero
    if params_to_keep is not None: 
        for param_name, indices in params_to_keep: 
            param = model_params_map[param_name]
            if indices is None: 
                new_grad = new_grad_calc(grads_1[param_name], grads_2[param_name])
                param.grad.data.copy_(new_grad.data)
            else: 
                new_grad = new_grad_calc(grads_1[param_name][indices], grads_2[param_name][indices])
                param.grad[indices] = new_grad.to(device)
    else: 
        for param_name, param in model_params_map.items(): 
            if grads_1[param_name] is not None and grads_2[param_name] is not None: 
                new_grad = new_grad_calc(grads_1[param_name], grads_2[param_name])
                param.grad.data.copy_(new_grad.data)

def take_optim_step(optimizer, model_params_map, param_partition, grads_1, grads_2, 
                    device, k=10000, new_grad_calc=_minimize_grads_2): 
    if k is not None: # do param partitioning
        similarities = compute_similarities(param_partition, grads_1, grads_2)
        params_to_keep = find_parameters_to_keep(param_partition, similarities, k=k, return_target_indices=False)
        update_model_param_grads(optimizer, model_params_map, params_to_keep, grads_1, grads_2, device, new_grad_calc=new_grad_calc)
    else: 
        update_model_param_grads(optimizer, model_params_map, None, grads_1, grads_2, device, new_grad_calc=new_grad_calc)

    optimizer.step()

def do_an_mlm_backprop(model, input_ids, attention_mask, indices, target_tokens, vocab_size, device, model_name='', 
                        do_backprop=True, multiplier=None): 

    input_ids = input_ids.to(device)
    indices = indices.to(device)
    target_tokens = target_tokens.to(device)
    attention_mask = attention_mask.to(device)

    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    if 'roberta' not in model_name: 
        logits = outputs.prediction_logits
    else: 
        logits = outputs.logits

    indices = indices.unsqueeze(-1).repeat(1, vocab_size).unsqueeze(1)
    target_tokens = target_tokens.unsqueeze(-1)
    logits = logits.gather(1, indices)
    unsqueeze_later = logits.shape[0]==1
    logits = logits.squeeze()
    if unsqueeze_later: 
        logits = logits.unsqueeze(0)
    logits = logits.gather(1, target_tokens)

    if multiplier is not None: 
        logits = logits*multiplier

    final_output = logits.sum()

    if do_backprop: 
        final_output.backward()

    return final_output, logits

def do_an_lm_backprop(model, input_ids, attention_mask, labels, device): 
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)

    model.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    loss = -outputs.loss
    loss.backward()

    return loss

def _get_checkpoint_dir(epoch, dedupe=''): 
    # TODO: make this better, for now it's usable (i.e., it should work better for restarting during a middle epoch)
    return f'models/{dedupe}/model_{epoch}'

def save_model(model, tokenizer, epoch, dedupe=''): 
    output_dir = _get_checkpoint_dir(epoch, dedupe=dedupe)
    logger.info(f'Saving model at {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir

def retrain(model, tokenizer, optimizer, device, dataloader: data.DataLoader, batch_size: int, 
            is_mlm: bool = True, k: int = 10000, num_epochs: int = 5, 
            sim_batch_size: int = -1, new_grad_calc=_minimize_grads_2, proportion_dev=0.5, 
            do_dynamic_gradient_selection: bool=False, 
            agg_dim=-1, start_at_epoch: int = 0, dedupe='', model_name=''): 
    '''
    
    :param sim_batch_size: the number of examples to find the params to keep based off of. For simplicity, 
    this must be either a multiple of `batch_size`, -1 to denote that it should be same as batch_size, or None
    to denote using the full dataset (which needs not be a multiple of batch_size). 
    '''


    logger.info('retraining mlm')

    if sim_batch_size is -1: 
        sim_batch_size = batch_size

    if do_dynamic_gradient_selection: 
        new_grad_calc = _maximize_grads

    if sim_batch_size is not None and (sim_batch_size<=0 or sim_batch_size%batch_size!=0): 
        raise ValueError(f'Batch size for computing similarity is invalid: {sim_batch_size}')

    vocab_size = len(tokenizer)
    params_map = get_params_map(model)
    param_partition = create_param_partition(params_map, dim_to_agg=agg_dim)

    for epoch in range(start_at_epoch, num_epochs): 
        logger.info(f'On epoch {epoch+1}/{num_epochs}')
        model.train()

        curr_sim_batch_count = 0
        curr_disadvantaged_grads = None
        curr_advantaged_grads = None
        for (disadvantaged_seqs, advantaged_seqs), \
                (disadvantaged_attention_masks, advantaged_attention_masks), \
                inds, \
                (disadvantaged_target_tokens, advantaged_target_tokens), \
                (disadvantaged_labels, advantaged_labels) in tqdm(dataloader): 

            optimizer.zero_grad()

            if is_mlm: 
                disadv_logits = do_an_mlm_backprop(model, 
                                    disadvantaged_seqs, 
                                    disadvantaged_attention_masks, 
                                    inds, 
                                    disadvantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=not do_dynamic_gradient_selection)[1]
            else: 
                do_an_lm_backprop(model, 
                                    disadvantaged_seqs, 
                                    disadvantaged_attention_masks, 
                                    disadvantaged_labels, 
                                    device)
                raise NotSupportedError

            if not do_dynamic_gradient_selection: 
                disadvantaged_grads = get_all_model_grads(model) 

            if is_mlm: 
                adv_logits = do_an_mlm_backprop(model, 
                                    advantaged_seqs, 
                                    advantaged_attention_masks, 
                                    inds, 
                                    advantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=not do_dynamic_gradient_selection)[1]
            else: 
                do_an_lm_backprop(model, 
                                    advantaged_seqs, 
                                    advantaged_attention_masks, 
                                    advantaged_labels, 
                                    device)
                raise NotSupportedError

            if not do_dynamic_gradient_selection: 
                advantaged_grads = get_all_model_grads(model) 

            if do_dynamic_gradient_selection: 
                disadvantaged_actually_disadvantaged = disadv_logits<adv_logits
                multiplier = disadvantaged_actually_disadvantaged.float()*2-1 # if actually disadvantaged, 1, else, -1
                do_an_mlm_backprop(model, 
                                    disadvantaged_seqs, 
                                    disadvantaged_attention_masks, 
                                    inds, 
                                    disadvantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=True, 
                                    multiplier=multiplier)
                disadvantaged_grads = get_all_model_grads(model) 
                do_an_mlm_backprop(model, 
                                    advantaged_seqs, 
                                    advantaged_attention_masks, 
                                    inds, 
                                    advantaged_target_tokens, 
                                    vocab_size, 
                                    device, 
                                    model_name=model_name, 
                                    do_backprop=True, 
                                    multiplier=-multiplier)
                advantaged_grads = get_all_model_grads(model) 

            curr_disadvantaged_grads = accumulate_grad(curr_disadvantaged_grads, disadvantaged_grads)
            curr_advantaged_grads = accumulate_grad(curr_advantaged_grads, advantaged_grads)

            curr_sim_batch_count += batch_size
            
            if sim_batch_size is not None and curr_sim_batch_count>=sim_batch_size: 
                take_optim_step(optimizer, params_map, param_partition, curr_disadvantaged_grads, curr_advantaged_grads, 
                                device, k=k, new_grad_calc=new_grad_calc)
                curr_sim_batch_count = 0
                curr_disadvantaged_grads = None
                curr_advantaged_grads = None

        if sim_batch_size is None: 
            take_optim_step(optimizer, params_map, param_partition, curr_disadvantaged_grads, curr_advantaged_grads, 
                            device, k=k, new_grad_calc=new_grad_calc)

        saved_model_dir = save_model(model, tokenizer, epoch+1, dedupe=dedupe)

def main(model_path_or_name: str, num_epochs: int = 5, is_mlm: bool = True, k: int = 10000, 
        sim_batch_size: int = -1, use_advantaged_for_grad: bool=True, agg_input: bool=True, 
        proportion_dev=0.75, do_dynamic_gradient_selection: bool=False, 
        lr: float = 1e-5, momentum: float = 0.9, batch_size: int = 16, seed: int = 89793, 
        num_workers: int = 4, start_at_epoch: int = 0, dedupe=''): 
    logger.info(f'Seed is {seed}')
    set_random_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_mlm: 
        if 'roberta' in model_path_or_name: 
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, add_prefix_space=True)
        else: 
            tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_name, pad_token=PAD_TOKEN, mask_token=MASK_TOKEN)
        raise ValueError('No non-mlms currently')
    model = AutoModelForPreTraining.from_pretrained(model_path_or_name)
    model.resize_token_embeddings(len(tokenizer))

    model.train()
    model.to(device)

    dataset = WGDataset('../data/wg.tsv', '../data/wg_stats.tsv', tokenizer)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=WGDataset.collate_batch_creator(tokenizer), 
                                num_workers=num_workers, 
                                )
    optimizer = optim.SGD(model.parameters(), lr=lr)

    agg_dim = -1 if agg_input else -2
    new_grad_calc = _minimize_grads_2 if use_advantaged_for_grad else _maximize_grads_1

    logger.info('Retraining now')

    retrain(model, tokenizer, optimizer, device, dataloader, batch_size, is_mlm=is_mlm, k=k, 
            num_epochs=num_epochs, sim_batch_size=sim_batch_size, new_grad_calc=new_grad_calc, 
            proportion_dev=proportion_dev, do_dynamic_gradient_selection=do_dynamic_gradient_selection, 
            agg_dim=agg_dim, start_at_epoch=start_at_epoch, dedupe=dedupe, model_name=model_path_or_name)

if __name__=='__main__': 
    parser = argparse.ArgumentParser(description = 'This script uses prompts to sample sentences from an LM')
    parser.add_argument('-m', type=str, required=True, dest='model_path_or_name', help='path to the model or name of the model')
    parser.add_argument('-l', type=float, default=1e-5, dest='lr', help='learning rate')
    parser.add_argument('-k', type=int, default=10000, dest='k', help='the k in top k')
    parser.add_argument('--use-full-grad', dest='k', action='store_const', const=None, help='to use the full gradient (rather than k parts)') # if this and also -k are used then whichever is rightmost argument wins
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-n', type=int, default=5, dest='num_epochs', help='number of epochs to train for (total)')
    parser.add_argument('-b', type=int, default=16, dest='batch_size', help='batch size')
    parser.add_argument('--start-at', type=int, default=0, help='start at checkpoint epoch number (e.g., 1, and if training 5 epochs then 4 more epochs will be done)')
    parser.add_argument('--dedupe', type=str, default='', help='dedupe string (basically just the name of the experiment), models will be saved to `sim_checkpoints/{dedupe}/model_{epoch}`')
    parser.add_argument('--output-agg', dest='aggregation', action='store_const', const='output', default='input', help='to use output aggregation (default: input aggregation)')
    parser.add_argument('--dynamic_gradient_selection', dest='dynamic_gradient_selection', action='store_true', default=False, help='to choose disadvantaged and advantaged dynamically (default: static based on WG)')
    parser.add_argument('--use-disadvantaged', dest='use_advantaged_for_grad', action='store_false', default=True, help='to take gradient step to maximize disadvantaged (default: minimize advantaged)')
    parser.add_argument('--use-same-params', dest='sim_batch_size', action='store_const', const=None, default=-1, help='to use the same params each epoch (default: picks params each batch)')

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    is_mlm = 'bert' in args.model_path_or_name
    sim_batch_size = args.sim_batch_size
    use_advantaged_for_grad = args.use_advantaged_for_grad
    agg_input = args.aggregation=='input'
    if args.dynamic_gradient_selection: 
        direction_selection = 'dynamic'
    elif args.use_advantaged_for_grad: 
        direction_selection = 'adv'
    else: 
        direction_selection = 'disadv'

    if args.k is None: 
        partition_usage = 'full_grad'
    elif args.sim_batch_size is None: 
        partition_usage = 'all'
    else: 
        partition_usage = 'notall'

    dedupe_model_name = args.model_path_or_name.split('/')[-1]
    dedupe = f"{dedupe_model_name}/{partition_usage}/{'inp' if agg_input else 'outp'}/{direction_selection}/{args.lr}/64/{args.k}"
    main(args.model_path_or_name, num_epochs=args.num_epochs, is_mlm=is_mlm, k=args.k, 
        proportion_dev=0.5, do_dynamic_gradient_selection=args.dynamic_gradient_selection, 
        sim_batch_size=sim_batch_size, use_advantaged_for_grad=use_advantaged_for_grad, agg_input=agg_input, 
        lr=args.lr, momentum=args.momentum, batch_size=args.batch_size, start_at_epoch=args.start_at, dedupe=dedupe)
