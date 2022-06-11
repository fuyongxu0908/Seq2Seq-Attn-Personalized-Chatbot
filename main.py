#!/usr/bin/python
# -*- coding:utf8 -*-
import torch.optim
import math
from torch.optim import lr_scheduler
import random

from utils import _set_args
from data_utils import _get_dataset, ConvAI2
from model import Seq2Seq
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader


def valid(args, data_loader, model):
    metrics = {'correct_tokens': 0, 'loss': 0, 'num_tokens': 0, 'ppl': 0}
    with tqdm(total=len(data_loader)) as t:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data_loader, desc="Validing"):
                input, res, input_lens, res_lens = batch
                if args.device == 'cuda':
                    input = input.to(args.device)
                    res = res.to(args.device)

                out = model(input, res, input_lens=input_lens, res_lens=res_lens)
                # generated response
                _preds, scores, cand_preds = out[0], out[1], out[2]

                score_view = scores.view(-1, scores.size(-1))
                loss = criterion(score_view, res.view(-1))
                # save loss to metrics
                y_ne = res.ne(dialog_dict.tok2ind['__null__'])
                target_tokens = y_ne.long().sum().item()
                correct = ((res == _preds) * y_ne).sum().item()
                try:
                    t.set_postfix(loss='{:.6f}'.format(loss.item()), ppl='{:.6f}'.format(math.exp(loss.item())))
                except:
                    t.set_postfix(loss='{:.6f}'.format(loss.item()), ppl='{:.6f}'.format(float('inf')))

                metrics['correct_tokens'] += correct
                metrics['loss'] += loss.item()
                metrics['ppl'] += math.exp(loss.item())
                # metrics['ppl'] += math.exp(loss.item())
                metrics['num_tokens'] += target_tokens
                loss /= target_tokens
            print('eps: %d, Valid Loss: {%.7f}, Valid PPL: {%.3f} \n' % (
            eps, metrics['loss'] / len(data_loader), metrics['ppl'] / len(data_loader)))
            '''
            predicted
            output = model(xs=input, input_lens=input_lens)
            predictionss = output[0]
            '''
    return metrics['loss'] / len(data_loader), metrics['ppl'] / len(data_loader)


def test(args, data_loader, model):
    metrics = {'correct_tokens': 0, 'loss': 0, 'num_tokens': 0, 'ppl': 0}
    with tqdm(total=len(data_loader)) as t:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data_loader, desc="Testing"):
                input, res, input_lens, res_lens = batch
                if args.device == 'cuda':
                    input = input.to(args.device)
                    res = res.to(args.device)

                out = model(input, res, input_lens=input_lens, res_lens=res_lens)
                # generated response
                _preds, scores, cand_preds = out[0], out[1], out[2]

                score_view = scores.view(-1, scores.size(-1))
                loss = criterion(score_view, res.view(-1))
                # save loss to metrics
                y_ne = res.ne(dialog_dict.tok2ind['__null__'])
                target_tokens = y_ne.long().sum().item()
                correct = ((res == _preds) * y_ne).sum().item()

                metrics['correct_tokens'] += correct
                metrics['loss'] += loss.item()
                metrics['ppl'] += math.exp(loss.item())
                # metrics['ppl'] += math.exp(loss.item())
                metrics['num_tokens'] += target_tokens
                loss /= target_tokens
            print('eps: %d, Test Loss: {%.7f}, Test PPL: {%.3f} \n' % (
                eps, metrics['loss'] / len(data_loader), metrics['ppl'] / len(data_loader)))
            return metrics['ppl'] / len(data_loader)


def predict(args, data_loader, model, dialogdict):
    with tqdm(total=len(data_loader)) as t:
        with torch.no_grad():
            model.eval()
            for batch in tqdm(data_loader, desc="Predicting"):
                input, res, input_lens, res_lens = batch
                if args.device == 'cuda':
                    input = input.to(args.device)
                    res = res.to(args.device)

                    output = model(xs=input, ys=None, beam_size=args.beam_size, topk=args.topk, input_lens=input_lens)
                    predictions, cand_preds = output[0], output[2]

                    bsz = predictions.shape[0]
                    res = []
                    inputs = []
                    for b in range(bsz):
                        pred = predictions[b]
                        inputs.append(dialogdict._decode(input[b]))
                        res.append(dialogdict._decode(pred))
    for inp, re in zip(inputs, res):
        print('TEXT: ', ' '.join(inp))
        print('PREDICTION: ', ' '.join(re), '\n~')


def save_model(args, eps, model, optimizer, scheduler, ppl):
    model_state = {}
    model_state.update({'model': model.state_dict()})
    model_state.update({'epoch': eps})
    #model_state.update({'optimizer': optimizer.state_dict()})
    model_state.update({'scheduler': scheduler.state_dict()})
    torch.save(model_state, args.save_path + '/{}_model_{}.pth'.format(str(eps), str(ppl)))


def load_model(model_path):
    model_state = torch.load(model_path)
    model_dict = model_state['model']
    eps = model_state['epoch']
    #optimizer_dict = model_state['optimizer']
    scheduler_dict = model_state['scheduler']
    #return eps, model_dict, optimizer_dict, scheduler_dict
    return eps, model_dict, scheduler_dict


if __name__ == '__main__':
    args = _set_args()
    datasets, dialog_dict = _get_dataset(args)
    train_dataset, valid_dataset = ConvAI2(datasets['train']), ConvAI2(datasets['valid'])
    train_loader, valid_loader = DataLoader(dataset=train_dataset, batch_size=args.train_bs, shuffle=True, collate_fn=lambda x: train_dataset.collate_fn(x, dialog_dict)), \
                                 DataLoader(dataset=valid_dataset[:7021], batch_size=args.valid_bs, shuffle=False, collate_fn=lambda x: train_dataset.collate_fn(x, dialog_dict))
    '''
    test_ind = random.sample(range(len(valid_dataset)), 10)
    test_set = [valid_dataset[i] for i in test_ind]
    test_loader = DataLoader(dataset=test_set, batch_size=10, shuffle=False, collate_fn=lambda x: train_dataset.collate_fn(x, dialog_dict))
    '''

    model = Seq2Seq(
        args=args,
        num_features=len(dialog_dict.ind2tok),
        padding_idx=dialog_dict.tok2ind['__null__'],
        start_idx=dialog_dict.tok2ind['__start__'],
        end_idx=dialog_dict.tok2ind['__end__']
    )

    criterion = nn.CrossEntropyLoss(ignore_index=dialog_dict.tok2ind['__null__'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    if args.load_from_path:
        print('Loding from ' + args.load_from_path)
        eps, model_dict, scheduler_dict = load_model(args.load_from_path)
        start_epoch = eps + 1
        model.load_state_dict(model_dict)
        #optimizer.load_state_dict(optimizer_dict)
        scheduler.load_state_dict(scheduler_dict)
        end_epoch = args.epochs
    else:
        start_epoch = 0
        end_epoch = args.epochs

    if args.device == 'cuda':
        model = model.cuda()
        criterion = criterion.cuda()

    for eps in range(start_epoch, end_epoch):
        metrics = {'correct_tokens': 0, 'loss': 0, 'num_tokens': 0, 'ppl': 0}

        with tqdm(total=len(train_loader)) as t:
            t.set_description('Epoch: {}/{}'.format(eps + 1, end_epoch))
            for batch in tqdm(train_loader, desc="Training"):
                input, res, input_lens, res_lens = batch
                if args.device == 'cuda':
                    input = input.to(args.device)
                    res = res.to(args.device)
                model.train()
                optimizer.zero_grad()
                out = model(input, res, input_lens=input_lens, res_lens=res_lens)
                # generated response
                _preds, scores, cand_preds = out[0], out[1], out[2]

                score_view = scores.view(-1, scores.size(-1))
                loss = criterion(score_view, res.view(-1))
                # save loss to metrics
                y_ne = res.ne(dialog_dict.tok2ind['__null__'])
                target_tokens = y_ne.long().sum().item()
                correct = ((res == _preds) * y_ne).sum().item()
                try:
                    t.set_postfix(loss='{:.6f}'.format(loss.item()), ppl='{:.6f}'.format(math.exp(loss.item())))
                except:
                    t.set_postfix(loss='{:.6f}'.format(loss.item()), ppl='{:.6f}'.format(float('inf')))

                metrics['correct_tokens'] += correct
                metrics['loss'] += loss.item()
                metrics['ppl'] += math.exp(loss.item())
                # metrics['ppl'] += math.exp(loss.item())
                metrics['num_tokens'] += target_tokens
                loss /= target_tokens
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

        print('eps: %d, Train Loss: {%.7f}, Train PPL: {%.3f} \n' % (eps, metrics['loss']/len(train_loader), metrics['ppl']/len(train_loader)))
        valid_loss, valid_ppl = valid(args, valid_loader, model)
        scheduler.step(valid_loss)
        predict(args, valid_loader, model=model, dialogdict=dialog_dict)
        #ppl = test(args, valid_loader, model)
        save_model(args, eps, model, optimizer, scheduler, valid_ppl)


