import torch
import torch.utils.data as data
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
import logging
import os
import json
from tqdm import tqdm
from dataset import StereoSetDataset
from eval_on_stereoset import compute_stereoset_scores, eval_mlm_given_dataloader

logger = logging.getLogger(__name__)

DOMAINS = ['gender', 'profession', 'race', 'religion']

def main(model_type, model_name, ss_dev_proportion=0.5, models_loc=f'./models', pretrained_only=False, shuffled=True): 
    dedupe = f"{ss_dev_proportion}"
    if not shuffled: 
        dedupe += "_unsh"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'roberta' in model_type: 
        tokenizer = AutoTokenizer.from_pretrained(model_type, add_prefix_space=True)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_type)

    logger.info(f'Initialized tokenizer')
    results_map = construct_results_map(model_name, tokenizer, ss_dev_proportion, device, models_loc=models_loc, pretrained_only=pretrained_only, shuffled=shuffled)
    if pretrained_only: 
        save_result(results_map, f'results/pretrained_{model_name}_results_map.json', dedupe=dedupe)
        return
    save_result(results_map, f'results/{model_name}_results_map.json', dedupe=dedupe)
    logger.info(f'Got results map {json.dumps(results_map, indent=4)}')
    best_results = best_results_per_level(results_map)
    logger.info(f'Got best results {json.dumps(best_results, indent=4)}')
    all_results_list = remove_final_level_results(best_results)
    all_results = {result['path']: result for result in all_results_list}
    logger.info(f'Got all results {json.dumps(all_results, indent=4)}')

    save_result(best_results, f'results/best_{model_name}_results.json', dedupe=dedupe)
    save_result(all_results, f'results/all_{model_name}_results.json', dedupe=dedupe)

def construct_ss_dataloaders(tokenizer, ss_dev_proportion, shuffled=True): 

    batch_size = 63

    dataloader_map = {}
    for domain in DOMAINS: 
        dataloader_map[domain] = {}

        dataset_path = f'../data/stereoset_eval/{domain}{"" if shuffled else "_unsh"}.txt'

        dev_dataset = StereoSetDataset(dataset_path, tokenizer, dev=True, proportion_dev=ss_dev_proportion)
        dev_dataloader = data.DataLoader(dev_dataset, batch_size=batch_size, 
                                        collate_fn=StereoSetDataset.collate_batch_creator(tokenizer))
        test_dataset = StereoSetDataset(dataset_path, tokenizer, dev=False, proportion_dev=ss_dev_proportion)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, 
                                        collate_fn=StereoSetDataset.collate_batch_creator(tokenizer))

        dataloader_map[domain]['dev'] = dev_dataloader
        dataloader_map[domain]['test'] = test_dataloader
        dataloader_map[domain]['dev_count'] = len(dev_dataset)
        dataloader_map[domain]['test_count'] = len(test_dataset)
        
    return dataloader_map

def construct_results_map(model_name, tokenizer, ss_dev_proportion, device, models_loc, pretrained_only=False, shuffled=True): 
    base_dir = f'{models_loc}/{model_name}/'

    if not pretrained_only: 
        model_paths = [
            x[0]
            for x in os.walk(base_dir) 
            if not x[1] # only leaves
        ]
    else: 
        model_paths = [model_name]

    logger.info(f'Found {len(model_paths)} models to evaluate.')
    logger.debug(model_paths)

    logger.info('Constructing StereoSet dataloaders')
    
    dataloader_map = construct_ss_dataloaders(tokenizer, ss_dev_proportion, shuffled=shuffled)

    logger.info('Constructed StereoSet dataloaders')

    # construct tree of hyperparams
    results_map = {}

    for i, model_path in enumerate(tqdm(model_paths)): 
        logger.info(f'Evaluating model at {model_path}')
        logger.info('='*50 + f'{i+1}/{len(model_paths)}' + '='*50)

        model = AutoModelForMaskedLM.from_pretrained(model_path)
        model.eval()
        model.to(device)

        result = {
            'path' : model_path, 
        }

        for domain, dataloaders in dataloader_map.items(): 
            ss_dev_dataloader = dataloaders['dev']
            ss_test_dataloader = dataloaders['test']

            result[domain] = {
                'dev' : generate_ss_result(model, tokenizer, device, ss_dev_dataloader), 
                'test' : generate_ss_result(model, tokenizer, device, ss_test_dataloader), 
                'dev_count': dataloaders['dev_count'], 
                'test_count': dataloaders['test_count'],
            }

        logger.info(f'Got result: {json.dumps(result, indent=4)}')
        logger.info('='*100)

        curr_map = results_map
        if not pretrained_only: 
            model_name = os.path.basename(model_path)
            split_path = os.path.split(model_path[len(base_dir):])[0].split('/')
            for hyperparam in split_path: 
                if hyperparam not in curr_map: 
                    curr_map[hyperparam] = {}
                curr_map = curr_map[hyperparam]
        curr_map[model_name] = result
    
    return results_map
 
def generate_ss_result(model, tokenizer, device, ss_dataloader): 
    results = eval_mlm_given_dataloader(model, tokenizer, device, ss_dataloader)
    ss_score, lms_score, icat_score = compute_stereoset_scores(*results)
    return {
        'ss': ss_score, 
        'lms': lms_score, 
        'icat': icat_score, 
    }   

def default_result(): 
    return {
        'best_model': None
    }

def score_result(result, stage): 
    unnormalized_score = 0
    total_count = 0
    domains = ['gender', 'profession']
    for domain in domains: 
        unnormalized_score += result[domain][stage]['ss']*result[domain][f'{stage}_count']
        total_count += result[domain][f'{stage}_count']
    score = unnormalized_score/total_count
    return -score # wrong version but it's what was used for most, fix result outputs eventually..
    # return -abs(score-0.5) # correct version

def result_better_than(result_1, result_2, scorer=score_result): # return result_1 > result_2
    scores = []
    for result in [result_1, result_2]: 
        if result is None: 
            scores.append(-float('inf'))
            continue
        if 'dev_criterion_score' not in result: 
            for stage in ['dev', 'test']: 
                score = scorer(result, stage)
                result[f'{stage}_criterion_score'] = score
        scores.append(result['dev_criterion_score'])
    return scores[0] > scores[1]

def best_results_per_level(results_map, key_name='', scorer=score_result): 
    '''
    Essentially, see the example at the very end of the file
    '''

    # leaf
    if 'model_' in key_name:
        return {
            'best_model': results_map, 
        }
    else:
        output_map = default_result()
        for child in results_map:
            child_best = best_results_per_level(results_map[child], key_name=child, scorer=scorer)
            output_map[child] = child_best
            if result_better_than(child_best['best_model'], output_map['best_model'], scorer=scorer): 
                output_map['best_model'] = child_best['best_model']
        return output_map

def remove_final_level_results(best_results_map, removed_results=[]): 
    to_remove_keys = []
    for child, child_map in best_results_map.items(): 
        if child=='best_model': 
            continue
        elif 'model_' in child: 
            to_remove_keys.append(child)
        else: 
            remove_final_level_results(child_map, removed_results=removed_results)
    for child in to_remove_keys: 
        removed_results.append(best_results_map.pop(child)['best_model'])
    
    return removed_results

def save_result(result, output_file, dedupe=None): 
    import os
    if dedupe: 
        file_name = f'{dedupe}{output_file}'
    else: 
        file_name = f'{output_file}'
    os.makedirs(os.path.split(file_name)[0], exist_ok=True)

    with open(file_name, 'w') as f: 
        json.dump(result, f, indent=4)

if __name__=='__main__': 
    MODEL_CHOICES = [
        'bert-base-cased', 
        'bert-base-uncased', 
        'albert-base-v2', 
        'roberta-base', 
    ]
    parser = argparse.ArgumentParser(description = 'This script uses prompts to sample sentences from an LM')
    parser.add_argument('-t', type=str, required=True, dest='model_type', choices=MODEL_CHOICES, help='the model type to evaluate')
    parser.add_argument('-m', type=str, required=True, dest='model_name', help='the model "name" to evaluate')
    parser.add_argument('--ss_dev_proportion', type=float, default=0.5, help='proportion of the original StereoSet dev set to use as our dev set')
    parser.add_argument('--models_base_loc', type=str, default='./models', help='relative path to the root of all the model checkpoints')
    parser.add_argument('--pretrained_only', dest='pretrained_only', action='store_true', default=False, help='to only evaluate the pretrained model (default: evaluate only all tuned models)')
    parser.add_argument('--unshuffled', dest='shuffled', action='store_false', default=True, help='to use unshuffled data (default: uses shuffled data)')

    args = parser.parse_args()

    log_filename = f"logs/evaluate_models_{args.model_name}.log" if not args.pretrained_only else f"logs/evaluate_pretrained_{args.model_name}.log"
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_filename), 
            # logging.StreamHandler(sys.stdout), 
        ],
    )
    
    main(args.model_type, args.model_name, ss_dev_proportion=args.ss_dev_proportion, models_loc=args.models_base_loc, pretrained_only=args.pretrained_only, shuffled=args.shuffled)