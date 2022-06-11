#!/usr/bin/python
# -*- coding:utf8 -*-
import json
import os

import torch
import tqdm
from transformers import cached_path
import codecs
from collections import defaultdict

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"


class DialogDict(object):
    def __init__(self, corpus_path, args):
        self.corpus_path = corpus_path
        self.ind2tok = {}
        self.tok2ind = {}
        self.freq = defaultdict(int)
        self.__null__ = '__null__'
        self.__start__ = '__start__'
        self.__end__ = '__end__'
        self.__unk__ = '__unk__'
        self.args = args

    def escape(self, s):
        r"""Replace potential special characters with escaped version.

        For example, \n => \\n and \t => \\t

        :param s: string to escape
        """
        return s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')

    def unescape(self, s):
        r"""Revert escaped characters back to their special version.

        For example, \\n => \n and \\t => \t

        :param s: string to unescape
        """
        return s.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')

    def _tokenize(self, text):
        return text.replace('.', ' . ') \
               .replace(',', ' , ') \
               .replace(';', ' ; ') \
               .replace(':', ' : ') \
               .replace('!', ' ! ') \
               .replace('?', ' ? ') \
               .replace(':', ' : ') \
               .split()

    def add_token(self, word):
        if word not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[word] = index
            self.ind2tok[index] = word

    def add_to_dict(self, tokens):
        for token in tokens:
            self.add_token(token)
            self.freq[token] += 1

    def build(self):
        self.tok2ind.update({'__null__': 0, '__start__': 1, '__end__': 2, '__unk__': 3})
        self.ind2tok.update({0: '__null__', 1: '__start__', 2: '__end__', 3: '__unk__'})
        self.freq[self.__null__] = 1000000003
        self.freq[self.__start__] = 1000000002
        self.freq[self.__end__] = 1000000001
        self.freq[self.__unk__] = 1000000000

        with codecs.open(self.corpus_path, 'r', 'utf8') as f:
            for line in f.readlines():
                self.add_to_dict(self._tokenize(line))

    def save(self):
        file_name = self.args.cache + 'Dialogue.dict'
        print('Dictionary: saving dictionary to {}'.format(file_name))
        with open(file_name, 'a') as write:
            for i in range(len(self.ind2tok)):
                tok = self.ind2tok[i]
                cnt = self.freq[tok]
                write.write('{tok}\t{cnt}\n'.format(tok=self.escape(tok), cnt=cnt))

    def load(self):
        file_name = self.args.cache + 'Dialogue.dict'
        print('Dictionary: loading dictionary from {}'.format(file_name))
        lower_special = self.__null__ == self.__null__.lower()
        SPECIAL_TOKENS = {'__unk__', '__null__', '__end__', '__start__'}
        with codecs.open(file_name, 'r', encoding='utf-8', errors='ignore') as read:
            for line in read:
                split = line.strip().split('\t')
                token = self.unescape(split[0])
                if lower_special and token in SPECIAL_TOKENS:
                    token = token.lower()
                cnt = int(split[1]) if len(split) > 1 else 0
                self.freq[token] = cnt
                self.add_token(token)
        print('[ num words =  %d ]' % len(self.ind2tok))

    def _decode(self, tokens_list):
        return [self.ind2tok[w.item()] for w in tokens_list]

def _get_dataset(args):
    corpus_path = args.cache + 'corpus.txt'
    datasetpath = args.cache + 'dataset.json'
    dic_path = args.cache + 'Dialogue.dict'

    if not os.path.exists(corpus_path) or not os.path.exists(datasetpath) or not os.path.exists(dic_path):
        dataset_path = args.dataset_path or PERSONACHAT_URL
        print("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        corpus = codecs.open(corpus_path, 'a', 'utf8')
        datasets = {"train": [], "valid": []}
        for dataset_name, data in dataset.items():
            for dialog in tqdm.tqdm(data, desc="Process Data"):
                persona = dialog['personality'].copy()
                for utter in dialog['utterances']:
                    query = utter['history'][-1]
                    datasets[dataset_name].append(('persona : ' + ' '.join(persona), query, utter['candidates'][-1]))
                    corpus.write('persona : ' + ' '.join(persona) + ' ' + query + ' ' + utter['candidates'][-1] + '\n')
        corpus.close()
        print("Save Datasets...")
        with open(datasetpath, 'w') as f:
            json.dump(datasets, f)
        dialog_dict = DialogDict(corpus_path, args)
        dialog_dict.build()
        dialog_dict.save()
    else:
        dialog_dict = DialogDict(corpus_path, args)
        dialog_dict.load()
        with open(datasetpath, 'r') as f:
            datasets = json.load(f)

    return datasets, dialog_dict


class ConvAI2(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def _tokenized(self, seq, dialog_dict):
        ret = []
        for w in seq.split():
            try:
                ret.append(dialog_dict.tok2ind[w])
            except:
                ret.append(dialog_dict.tok2ind['__unk__'])
        return ret

    def collate_fn(self, batch, dialog_dict):
        input_seq = [dialog_dict.__start__ + ' ' + b[0] + ' ' + b[1] for b in batch]
        response_seq = [b[2] + ' ' + dialog_dict.__end__ for b in batch]

        assert len(input_seq) == len(response_seq)
        input_seq_tokenized = [self._tokenized(seq, dialog_dict) for seq in input_seq]
        response_seq_tokenized = [self._tokenized(res, dialog_dict) for res in response_seq]

        sort_map = [(i, j) for i, j in zip(input_seq_tokenized, response_seq_tokenized)]
        sorted_map = sorted(sort_map, key=lambda x: len(x[0]), reverse=True)

        input_lens = [len(it[0]) for it in sorted_map]
        res_lens = [len(it[1]) for it in sorted_map]

        input = pad_sequence([torch.tensor(it[0]) for it in sorted_map], batch_first=True, padding_value=dialog_dict.tok2ind[dialog_dict.__null__])
        res = pad_sequence([torch.tensor(it[1]) for it in sorted_map], batch_first=True, padding_value=dialog_dict.tok2ind[dialog_dict.__null__])

        return input, res, input_lens, res_lens
