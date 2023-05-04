from typing import List, Tuple
import os
import csv
import torch
import torch.nn as nn
import torch.utils.data as data
import math
from consts import PAD_VALUE

def save_sampled_sents(filename: str, sents: List[str]): 
    dir = os.path.split(filename)[0]
    if dir: 
        os.makedirs(dir, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f: 
        for sent in sents: 
            if sent: 
                f.write(sent)
                f.write('\n')

def load_sampled_sents(filename: str) -> List[str]: 
    with open(filename, 'r', encoding='utf-8') as f: 
        sents = [line.strip() for line in f if line]

    return sents

def load_sampled_sents_2(filename: str) -> List[Tuple]: 
    with open(filename, 'r', encoding='utf-8') as f: 
        examples = [tuple(line.strip().split('\t')) for line in f if line]

    return examples 

def load_wg_stats(filename): 
    pcts = {}
    with open(filename, 'r', encoding='utf-8') as f: 
        for line in f.readlines()[1:]: 
            if line: 
                split_line = line.strip().split('\t')
                pcts[split_line[0]] = float(split_line[1]) # using the bergsma proportion
    return pcts

def load_sents_wg(filename: str) -> List[Tuple]: 
    sents = []
    occs = []
    with open(filename, 'r', encoding='utf-8') as f: 
        for line in f.readlines()[1:]: 
            if line: 
                split_line = line.strip().split('\t')
                sents.append(split_line[1])
                occs.append(split_line[0].split('.')[0])

    sent_triples = []
    trip_occ = []
    for i in range(0, len(sents), 3): 
        sent_triples.append((sents[i], sents[i+1], sents[i+2]))
        trip_occ.append(occs[i])
    
    return sent_triples, trip_occ

def load_sents_crows(filename: str): 
    with open(filename, 'r', encoding='utf-8', newline='') as csvfile: 
        csv_reader = csv.reader(csvfile)
        lines = list(csv_reader)
    sent_pairs = [(row[1], row[2]) for row in lines]
    return sent_pairs

class StereoSetDataset(data.Dataset): 
    def __init__(self, dataset_file, tokenizer, max_len=64, dev=True, proportion_dev=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        examples = load_sampled_sents_2(dataset_file)
        
        tokenized_mask = tokenizer.mask_token_id
        tokenized_unk = tokenizer.unk_token_id

        sents, target_words = [list(unzipped) for unzipped in zip(*examples)]

        tokenized_sents = [tokenizer(sent, add_special_tokens=False)['input_ids'][:max_len] for sent in sents]
        tokenized_target_words = [tokenizer(target, add_special_tokens=False)['input_ids'] for target in target_words]

        inds = [-1]*len(tokenized_target_words)
        for i, (tokenized_sent, tokenized_target_word) in enumerate(zip(tokenized_sents, tokenized_target_words)): 
            if len(tokenized_target_word)!=1: 
                inds[i] = -2
                continue
            
            tokenized_target = tokenized_target_word[0]

            if tokenized_target==tokenized_unk: 
                continue

            token_indices = [i for i, token in enumerate(tokenized_sent) if token==tokenized_target]
            if len(token_indices)!=1: 
                inds[i] = -3
                continue
            
            inds[i] = token_indices[0]
        
        for i in range(2, len(inds), 3): 
            if inds[i]<0 or inds[i-1]<0 or inds[i-2]<0: 
                inds[i] = -4
                inds[i-1] = -4
                inds[i-2] = -4


        kept_sents = [tokenized_sent for i, tokenized_sent in enumerate(tokenized_sents) if inds[i]>=0]
        kept_inds = [ind for ind in inds if ind>=0]
        kept_target_tokens = [tokens[0] for i, tokens in enumerate(tokenized_target_words) if inds[i]>=0]

        for sent, ind in zip(kept_sents, kept_inds): 
            sent[ind] = tokenized_mask

        examples = list(zip(kept_sents, kept_inds, kept_target_tokens))
        example_count = len(examples)//3
        if example_count*3!=len(examples): 
            raise ValueError('malformed stereoset input file?')
        cutoff = math.floor(example_count * proportion_dev)*3

        if dev: 
            self.examples = examples[:cutoff]
        else: 
            self.examples = examples[cutoff:]

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer): 
        def collate_batch(batch): 
            sents, inds, target_tokens = [list(unzipped) for unzipped in zip(*batch)]

            seqs = [torch.tensor(sent) for sent in sents]
            inds = torch.LongTensor(inds)
            target_tokens = torch.LongTensor(target_tokens)

            padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_masks = torch.ones_like(padded_seqs)
            attention_masks[padded_seqs==tokenizer.pad_token_id] = 0
            labels = padded_seqs.detach().clone()
            labels[padded_seqs==tokenizer.pad_token_id] = PAD_VALUE

            return padded_seqs, attention_masks, inds, target_tokens, labels
        return collate_batch

class CrowsDataset(data.Dataset): 
    def __init__(self, dataset_file, tokenizer, max_len=24):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        examples = load_sents_crows(dataset_file)
        
        tokenized_mask = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]
        self.tokenized_unk = tokenizer('[UNK]', add_special_tokens=False)['input_ids'][0]

        sent_pairs = []
        inds = []
        target_token_pairs = []
        skipped_count = 0
        for first_sent, second_sent in examples: 
            ind, (t_first, first), (t_second, second) = self._tokenize_and_find_diff(first_sent, second_sent)
            if ind is None: 
                skipped_count += 1
                continue
            sent_pairs.append((t_first, t_second))
            inds.append(ind)
            target_token_pairs.append((first, second))
        print(f'Skipped {skipped_count}/{len(examples)}')

        for (first, second), ind in zip(sent_pairs, inds): 
            first[ind] = tokenized_mask
            second[ind] = tokenized_mask

        self.examples = list(zip(sent_pairs, inds, target_token_pairs))

    def _tokenize_and_find_diff(self, first, second): 
        tokenized_first = self.tokenizer(first, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_second = self.tokenizer(second, add_special_tokens=False)['input_ids'][:self.max_len]

        if len(tokenized_first)==len(tokenized_second): 
            found_i, found_f, found_s = None, None, None
            for i, (f, s) in enumerate(zip(tokenized_first, tokenized_second)): 
                if f==s: 
                    continue
                else: 
                    if found_i is not None: # only for single token diffs
                        return None, (None, None), (None, None)
                    found_i = i
                    found_f = f
                    found_s = s
            if found_i is None or found_f==self.tokenized_unk or found_s==self.tokenized_unk: 
                return None, (None, None), (None, None)
            return i, (tokenized_first, found_f), (tokenized_second, found_s)
        return None, (None, None), (None, None)

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer): 
        def collate_batch(batch): 
            sent_pairs, inds, target_token_pairs = [list(unzipped) for unzipped in zip(*batch)]

            seqs = []
            for first_sent, second_sent in sent_pairs: 
                seqs.append(torch.tensor(first_sent))
                seqs.append(torch.tensor(second_sent))
            full_inds = []
            for ind in inds: 
                full_inds.append(ind)
                full_inds.append(ind)
            inds = torch.LongTensor(full_inds)
            target_tokens = []
            for first_target, second_target in target_token_pairs: 
                target_tokens.append(first_target)
                target_tokens.append(second_target)
            target_tokens = torch.LongTensor(target_tokens)

            padded_seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_masks = torch.ones_like(padded_seqs)
            attention_masks[padded_seqs==tokenizer.pad_token_id] = 0
            
            labels = padded_seqs.detach().clone()
            labels[padded_seqs==tokenizer.pad_token_id] = PAD_VALUE
            
            return padded_seqs, attention_masks, inds, target_tokens, labels
        return collate_batch

class WGDataset(data.Dataset): 
    def __init__(self, dataset_file, stats_file, tokenizer, max_len=24):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        sent_triples, occupations = load_sents_wg(dataset_file)
        stat_pcts = load_wg_stats(stats_file)
        
        tokenized_mask = tokenizer('[MASK]', add_special_tokens=False)['input_ids'][0]

        sent_pairs = []
        inds = []
        target_token_pairs = []
        for sent_trip, occ in zip(sent_triples, occupations): 
            male, female, neutral = sent_trip
            ind, (t_male, m), (t_female, f), (t_neutral, n) = self._tokenize_and_find_diff(male, female, neutral)
            if ind is None: 
                continue
            if stat_pcts[occ]<50: 
                disadvantaged = t_female
                disadvantaged_token = f
                advantaged = t_male
                advantaged_token = m
            else: 
                disadvantaged = t_male
                disadvantaged_token = m
                advantaged = t_female
                advantaged_token = f
            sent_pairs.append((disadvantaged, advantaged))
            inds.append(ind)
            target_token_pairs.append((disadvantaged_token, advantaged_token))

        for (disadvantaged, advantaged), ind in zip(sent_pairs, inds): 
            disadvantaged[ind] = tokenized_mask
            advantaged[ind] = tokenized_mask

        self.examples = list(zip(sent_pairs, inds, target_token_pairs))
    
    def _tokenize_and_find_diff(self, male, female, neutral): 
        tokenized_male = self.tokenizer(male, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_female = self.tokenizer(female, add_special_tokens=False)['input_ids'][:self.max_len]
        tokenized_neutral = self.tokenizer(neutral, add_special_tokens=False)['input_ids'][:self.max_len]

        if len(tokenized_male)==len(tokenized_female)==len(tokenized_neutral): 
            for i, (m, f, n) in enumerate(zip(tokenized_male, tokenized_female, tokenized_neutral)): 
                if m==f==n: 
                    continue
                else: 
                    return i, (tokenized_male, m), (tokenized_female, f), (tokenized_neutral, n)
        return None, None, None, None

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def collate_batch_creator(tokenizer): 
        def collate_batch(batch): 
            sent_pairs, inds, target_token_pairs = [list(unzipped) for unzipped in zip(*batch)]
            disadvantaged_sents, advantaged_sents = [list(unzipped) for unzipped in zip(*sent_pairs)]
            disadvantaged_targets, advantaged_targets = [list(unzipped) for unzipped in zip(*target_token_pairs)]

            disadvantaged_seqs = [torch.tensor(sent) for sent in disadvantaged_sents]
            advantaged_seqs = [torch.tensor(sent) for sent in advantaged_sents]
            inds = torch.LongTensor(inds)
            disadvantaged_target_tokens = torch.LongTensor(disadvantaged_targets)
            advantaged_target_tokens = torch.LongTensor(advantaged_targets)

            padded_disadvantaged_seqs = nn.utils.rnn.pad_sequence(disadvantaged_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            padded_advantaged_seqs = nn.utils.rnn.pad_sequence(advantaged_seqs, batch_first=True, padding_value=tokenizer.pad_token_id)
            disadvantaged_attention_masks = torch.ones_like(padded_disadvantaged_seqs)
            advantaged_attention_masks = torch.ones_like(padded_advantaged_seqs)
            disadvantaged_attention_masks[padded_disadvantaged_seqs==tokenizer.pad_token_id] = 0
            advantaged_attention_masks[padded_advantaged_seqs==tokenizer.pad_token_id] = 0
            
            disadvantaged_labels = padded_disadvantaged_seqs.detach().clone()
            disadvantaged_labels[padded_disadvantaged_seqs==tokenizer.pad_token_id] = PAD_VALUE
            advantaged_labels = padded_advantaged_seqs.detach().clone()
            advantaged_labels[padded_advantaged_seqs==tokenizer.pad_token_id] = PAD_VALUE
            
            return (padded_disadvantaged_seqs, padded_advantaged_seqs), \
                    (disadvantaged_attention_masks, advantaged_attention_masks), \
                    inds, \
                    (disadvantaged_target_tokens, advantaged_target_tokens), \
                    (disadvantaged_labels, advantaged_labels)
        return collate_batch