import json
import os
import numpy as np

with open('./stereoset_dev.json', 'r') as f: 
    dataset = json.load(f)

BLANK = 'BLANK'
TARGET_BLANK = True
BIAS_TYPE = '' # 'gender', 'profession', 'race', 'religion', or empty/None to indicate all
SHUFFLE = True

LABELS_TO_KEEP = ['stereotype']
if TARGET_BLANK: 
    LABELS_TO_KEEP.append('anti-stereotype')
    LABELS_TO_KEEP.append('unrelated')

def extract_target_word_inds(context, target_word=BLANK): 
    context = context.upper()
    target_word = target_word.upper()
    # inefficient but whatever, 1 time thing only for stereoset
    for i in range(len(context)): 
        if context[i:i+len(target_word)] == target_word: 
            return (i, i+len(target_word))
    raise ValueError(f'Context sentence "{context}" does not contain "{target_word}"')

def extract_target_word(sentence, context, target_word_inds, target_blank=True): 
    start_ind = target_word_inds[0]
    end_ind = target_word_inds[1]-len(context)
    if end_ind==0: 
        end_ind = len(sentence)
    
    if target_blank: 
        target_word = sentence[start_ind:end_ind]
    else: 
        target_word = context[start_ind:end_ind]
        if ' ' in target_word and target_word not in sentence: 
            target_word_split = target_word.split(' ')
            target_word = ' '.join([target_word_split[0]] + [word_part.lower() for word_part in target_word_split[1:]])
        if target_word not in sentence: 
            target_word = target_word.lower()
        if target_word not in sentence: 
            target_word = target_word.capitalize()
        if target_word not in sentence: 
            print(f'Could not extract: {target_word} | {sentence} | {context}')
            target_word = None
    return target_word

all_items = dataset['data']['intrasentence']

indices = np.arange(len(dataset['data']['intrasentence']))
if SHUFFLE: 
    rng = np.random.default_rng(14159)
    rng.shuffle(indices)

could_not_extract_count = 0
all_sents = []
sents_per_domain = {
    'gender': [], 
    'profession': [], 
    'race': [], 
    'religion': [], 
}
for index in indices: 
    item = all_items[index]
    bias_type = item['bias_type']
    context = item['context']
    target_word = BLANK if TARGET_BLANK else item['target']
    target_word_inds = extract_target_word_inds(context, target_word=target_word)
    
    sentences = {}
    for sent_obj in item['sentences']: 
        sentences[sent_obj['gold_label']] = sent_obj['sentence']
    
    for label in LABELS_TO_KEEP: 
        sentence = sentences[label]
        target_word = extract_target_word(sentence, context, target_word_inds, target_blank=TARGET_BLANK)
        if target_word is None: 
            could_not_extract_count += 1
            continue
        all_sents.append((sentence,target_word))
        sents_per_domain[bias_type].append((sentence,target_word))

for sent, target in all_sents: 
    if not (sent or target): 
        raise ValueError

def write_sents_to_file(filename, sents): 
    with open(filename, 'w', encoding='utf-8') as f: 
        for sent, target in sents: 
            if sent and target: 
                f.write(sent)
                f.write('\t')
                f.write(target)
                f.write('\n')



write_sents_to_file(f'../data/stereoset_eval/all{"" if SHUFFLE else "_unsh"}.txt', all_sents)
for domain, sents in sents_per_domain.items(): 
    write_sents_to_file(f'../data/stereoset_eval/{domain}{"" if SHUFFLE else "_unsh"}.txt', sents)