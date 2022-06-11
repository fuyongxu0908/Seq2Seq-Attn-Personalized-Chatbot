#!/usr/bin/python
# -*- coding:utf8 -*-
import argparse
import torch
import numpy as np


def _set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sed', type=int, default=1234, help='The random seed.')
    parser.add_argument('--dataset_path', default='')
    parser.add_argument('--cache', type=str, default='./dataset_cache/')
    parser.add_argument('--tokenize', default='split')
    parser.add_argument('--train_bs', default=64)
    parser.add_argument('--valid_bs', default=32)
    parser.add_argument('--epochs', default=400)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--attn_type', default='general')
    parser.add_argument('--rnn_class', default='lstm')
    parser.add_argument('--lr', default=3)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--clip', default=0.1)
    parser.add_argument('--save_path', default='./ckp')
    parser.add_argument('--load_from_path', default='./ckp/1_model_194.99625568474823.pth')

    parser.add_argument('--embedding_size', default=256)
    parser.add_argument('--hidden_size', default=1024)
    parser.add_argument('--attn_length', default=48)
    parser.add_argument('--num_layers', default=2)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--attn_time', default='post')
    parser.add_argument('--bidirectional', default=True)
    parser.add_argument('--softmax_layer_bias', default=False)
    parser.add_argument('--num_softmax', default=1)
    parser.add_argument('--beam_size', default=1)
    parser.add_argument('--topk', default=1)

    args = parser.parse_args()
    return args


def _set_seed(args):
    np.random.seed(args.sed)
    torch.manual_seed(args.sed)
    torch.cuda.manual_seed_all(args.sed)
    np.random.seed(args.sed)
    torch.backends.cudnn.deterministic = True