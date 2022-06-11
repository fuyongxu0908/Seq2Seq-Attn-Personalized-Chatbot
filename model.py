#!/usr/bin/python
# -*- coding:utf8 -*-
from collections import deque, namedtuple
from operator import attrgetter
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import os


class Beam(object):
    """
    Generic beam class.

    It keeps information about beam_size hypothesis.
    """

    def __init__(
        self,
        beam_size,
        min_length=3,
        padding_token=0,
        bos_token=1,
        eos_token=2,
        min_n_best=3,
        cuda='cpu',
    ):
        """
        Instantiate Beam object.

        :param beam_size: number of hypothesis in the beam
        :param min_length: minimum length of the predicted sequence
        :param padding_token: Set to 0 as usual in ParlAI
        :param bos_token: Set to 1 as usual in ParlAI
        :param eos_token: Set to 2 as usual in ParlAI
        :param min_n_best: Beam will not be done unless this amount of finished
                           hypothesis (with EOS) is done
        :param cuda: What device to use for computations
        """
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = cuda
        # recent score for each hypo in the beam
        self.scores = torch.Tensor(self.beam_size).float().zero_().to(self.device)
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(padding_token).to(self.device)
        ]
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.HypothesisTail = namedtuple(
            'HypothesisTail', ['timestep', 'hypid', 'score', 'tokenid']
        )
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best

    def get_output_from_current_step(self):
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        voc_size = softmax_probs.size(-1)
        if len(self.bookkeep) == 0:
            # the first step we take only the first hypo into account since all
            # hypos are the same initially
            beam_scores = softmax_probs[0]
        else:
            # we need to sum up hypo scores and curr softmax scores before topk
            # [beam_size, voc_size]
            beam_scores = softmax_probs + self.scores.unsqueeze(1).expand_as(
                softmax_probs
            )
            for i in range(self.outputs[-1].size(0)):
                #  if previous output hypo token had eos
                # we penalize those word probs to never be chosen
                if self.outputs[-1][i] == self.eos:
                    # beam_scores[i] is voc_size array for i-th hypo
                    beam_scores[i] = -1e20

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        with torch.no_grad():
            best_scores, best_idxs = torch.topk(
                flatten_beam_scores, self.beam_size, dim=-1
            )

        self.scores = best_scores
        self.all_scores.append(self.scores)
        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.scores[hypid],
                    tokenid=self.eos,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Get single best hypothesis.

        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return (
            self.get_hyp_from_finished(top_hypothesis_tail),
            top_hypothesis_tail.score,
        )

    def get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep: timestep with range up to len(self.outputs)-1
        :param hyp_id: id with range up to beam_size-1
        :return: hypothesis sequence
        """
        assert self.outputs[hypothesis_tail.timestep][hypothesis_tail.hypid] == self.eos
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                self.HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback],
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def get_pretty_hypothesis(self, list_of_hypotails):
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)

        hypothesis = torch.stack(list(reversed(hypothesis)))

        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """

        :param n_best: how many n best hypothesis to return
        :return: list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, 0.65)
            rescored_finished.append(
                self.HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                )
            )

        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """
        Checks if self.finished is empty and add hyptail in that case.

        This will be suboptimal hypothesis since the model did not get any EOS

        :returns: None
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have eos
            # to pass assert in L102, it is ok since empty self.finished
            # means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(
                timestep=len(self.outputs) - 1,
                hypid=0,
                score=self.all_scores[-1][0],
                tokenid=self.outputs[-1][0],
            )

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """
        Creates pydot graph representation of the beam.

        :param outputs: self.outputs from the beam
        :param dictionary: tok 2 word dict to save words in the tree nodes
        :returns: pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type='digraph')
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ['aquamarine', 'chocolate1', 'deepskyblue', 'green2', 'tan']
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            # do not include EOS since it has rescored score not from original
            # self.all_scores, we color EOS with black
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(hyptail))

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(
                    timestep=tstep,
                    hypid=hypid,
                    score=all_scores[tstep][hypid],
                    tokenid=token,
                )
                color = 'white'
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = (
                    "<{}".format(
                        dictionary.vec2txt([token]) if dictionary is not None else token
                    )
                    + " : "
                    + "{:.{prec}f}>".format(all_scores[tstep][hypid], prec=3)
                )

                graph.add_node(
                    pydot.Node(
                        node_tail.__repr__(),
                        label=label,
                        fillcolor=color,
                        style='filled',
                        xlabel='{}'.format(rank) if rank is not None else '',
                    )
                )

        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node(
                    '"{}"'.format(
                        self.HypothesisTail(
                            timestep=revtstep,
                            hypid=prev_id,
                            score=all_scores[revtstep][prev_id],
                            tokenid=outputs[revtstep][prev_id],
                        ).__repr__()
                    )
                )[0]
                to_node = graph.get_node(
                    '"{}"'.format(
                        self.HypothesisTail(
                            timestep=revtstep + 1,
                            hypid=i,
                            score=all_scores[revtstep + 1][i],
                            tokenid=outputs[revtstep + 1][i],
                        ).__repr__()
                    )
                )[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph


class AttentionLayer(nn.Module):
    def __init__(
            self,
            attn_type,
            hidden_size,
            emb_size,
            bidirectional=False,
            attn_length=-1,
            attn_time='pre'
    ):
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hidden_size
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = emb_size
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')
            self.attn_combine = nn.Linear(hszXdirs + 2 * input_dim, input_dim, bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, enc_out, attn_mask=None):
        if self.attention == 'none':
            return xes

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        if self.attention == 'local':
            if enc_out.size(1) > self.max_length:
                offset = enc_out.size(1) - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)
            if attn_weights.size(1) > enc_out.size(1):
                attn_weights = attn_weights.narrow(1, 0, enc_out.size(1))
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                hid = hid.expand(
                    last_hidden.size(0), enc_out.size(1), last_hidden.size(1)
                )
                h_merged = torch.cat((enc_out, hid), 2)
                active = torch.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                if hid.size(2) != enc_out.size(2):
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                attn_w_premask = torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1)
            elif self.attention == 'general':
                hid = self.attn(hid)
                attn_w_premask = torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1)
            # calculate activation scores
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask -= (1 - attn_mask) * 1e20
            attn_weights = F.softmax(attn_w_premask, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        output = torch.tanh(self.attn_combine(merged).unsqueeze(1))
        return output


class Encoder(nn.Module):
    def __init__(
            self,
            num_features,
            padding_idx=0,
            rnn_class='lstm',
            emb_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            sparse=False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.emb_layer = nn.Embedding(
            num_features,
            emb_size,
            padding_idx=padding_idx,
            sparse=sparse)

        self.rnn = rnn_class(
            emb_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, xs, input_lens=None):
        bsz = len(xs)

        # embed input tokens
        xes = self.dropout(self.emb_layer(xs))
        xes = pack_padded_sequence(xes, input_lens, batch_first=True)
        encoder_output, hidden = self.rnn(xes)
        encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True)
        '''
        if self.bidirectional:
            if isinstance(self.rnn, nn.LSTM):
                hidden = (
                    hidden[0].view(-1, 2, bsz, self.hidden_size).max(1)[0],
                    hidden[1].view(-1, 2, bsz, self.hidden_size).max(1)[0],
                )# 2, bidirectional
            else:
                hidden = hidden.view(-1, 2, bsz, self.hidden_size).max(1)[0]
        '''
        return encoder_output, hidden


class Decoder(nn.Module):

    def __init__(
            self,
            num_features,
            padding_idx=0,
            rnn_class='lstm',
            emb_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidir_input=False,
            sparse=False,
            numsoftmax=1,
            softmax_layer_bias=False,
            attn_type='none',
            attn_length=-1,
            attn_time='pre'
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.hsz = hidden_size
        self.esz = emb_size

        self.embedd_layer = nn.Embedding(num_features, emb_size, padding_idx=padding_idx, sparse=sparse)
        self.rnn = rnn_class(emb_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidir_input)

        # rnn output to embedding
        if hidden_size != emb_size and numsoftmax == 1:
            # self.o2e = RandomProjection(hidden_size, emb_size)
            # other option here is to learn these weights
            self.o2e = nn.Linear(hidden_size, emb_size, bias=False)
        else:
            # no need for any transformation here
            self.o2e = lambda x: x
        # embedding to scores, use custom linear to possibly share weights
        self.e2s = nn.Linear(emb_size, num_features, bias=softmax_layer_bias)

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(
            attn_type=attn_type,
            hidden_size=hidden_size,
            emb_size=emb_size,
            bidirectional=bidir_input,
            attn_length=attn_length,
            attn_time=attn_time,
        )

        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.sofrmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hidden_size, numsoftmax, bias=False)
            self.latent = nn.Linear(hidden_size, numsoftmax * emb_size)
            self.activation = nn.Tanh()

    def forward(self, xs, hidden, encoder_output, attn_mask=None, topk=1):
        xes = self.dropout(self.embedd_layer(xs))
        if self.attn_time == 'pre':
            xes = self.attention(xes, hidden, encoder_output, attn_mask)
        if xes.dim() == 2:
            # if only one token inputted, sometimes needs unsquezing
            xes.unsqueeze_(1)
        output, new_hidden = self.rnn(xes, hidden)
        if self.attn_time == 'post':
            output = self.attention(output, new_hidden, encoder_output, attn_mask)

        if self.numsoftmax > 1:
            bsz = xs.size(0)
            seqlen = xs.size(1) if xs.dim() > 1 else 1
            latent = self.latent(output)
            active = self.dropout(self.activation(latent))
            logit = self.e2s(active.view(-1, self.esz))

            prior_logit = self.prior(output).view(-1, self.numsoftmax)
            prior = self.softmax(prior_logit)  # softmax over numsoftmax's

            prob = self.softmax(logit).view(bsz * seqlen, self.numsoftmax, -1)
            probs = (prob * prior.unsqueeze(2)).sum(1).view(bsz, seqlen, -1)
            scores = probs.log()

        else:
            e = self.dropout(self.o2e(output))
            scores = self.e2s(e)

            # select top scoring index, excluding the padding symbol (at idx zero)
            # we can do topk sampling from renoramlized softmax here, default topk=1 is greedy
            if topk == 1:
                _max_score, idx = scores.narrow(2, 1, scores.size(2) - 1).max(2)
            elif topk > 1:
                max_score, idx = torch.topk(
                    F.softmax(scores.narrow(2, 1, scores.size(2) - 1), 2),
                    topk,
                    dim=2,
                    sorted=False,
                )
                probs = F.softmax(
                    scores.narrow(2, 1, scores.size(2) - 1).gather(2, idx), 2
                ).squeeze(1)
                dist = torch.distributions.categorical.Categorical(probs)
                samples = dist.sample()
                idx = idx.gather(-1, samples.unsqueeze(1).unsqueeze(-1)).squeeze(-1)
            preds = idx.add_(1)

            return preds, scores, new_hidden


def pad(tensor, length, dim=0):
    if tensor.size(dim) < length:
        return torch.cat(
            [
                tensor,
                tensor.new(
                    *tensor.size()[:dim],
                    length - tensor.size(dim),
                    *tensor.size()[dim + 1 :],
                ).zero_(),
            ],
            dim=dim,
        )
    else:
        return tensor


class Seq2Seq(nn.Module):
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, args, num_features, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()
        self.args = args

        self.attn_type = args.attn_type
        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        rnn_class = Seq2Seq.RNN_OPTS[args.rnn_class]
        self.decoder = Decoder(
            num_features,
            padding_idx=self.NULL_IDX,
            rnn_class=rnn_class,
            emb_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            attn_type=args.attn_type,
            attn_length=args.attn_length,
            attn_time=args.attn_time,
            bidir_input=args.bidirectional,
            numsoftmax=args.num_softmax,
            softmax_layer_bias=args.softmax_layer_bias
        )
        self.encoder = Encoder(
            num_features,
            padding_idx=self.NULL_IDX,
            rnn_class=rnn_class,
            emb_size=self.args.embedding_size,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            bidirectional=self.args.bidirectional,

        )

    def forward(self, xs, ys=None, beam_size=1, topk=1, prev_enc=None, input_lens=None, res_lens=None):
        input_xs = xs
        nbest_beam_preds, nbest_beam_scores = None, None
        bsz = len(xs)
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        if prev_enc is not None:
            enc_out, hidden, attn_mask = prev_enc
        else:
            enc_out, hidden = self.encoder(xs, input_lens)
            attn_mask = xs.ne(0).float() if self.attn_type != 'none' else None
        encoder_states = (enc_out, hidden, attn_mask)
        start = self.START.detach()
        starts = start.expand(bsz, 1)

        predictions = []
        scores = []
        cand_preds, cand_scores = None, None

        if ys is not None:
            y_in = ys.narrow(1, 0, ys.size(1) - 1)
            xs = torch.cat([starts, y_in], 1)
            if self.attn_type == 'none':
                preds, score, hidden = self.decoder(xs, hidden, enc_out, attn_mask)
                predictions.append(preds)
                scores.append(score)
            else:
                for i in range(ys.size(1)):
                    xi = xs.select(1, i)
                    preds, score, hidden = self.decoder(xi, hidden, enc_out, attn_mask)
                    predictions.append(preds)
                    scores.append(score)
        else:
            # here we do search: supported search types: greedy, beam search
            if beam_size == 1:
                done = [False for _ in range(bsz)]
                total_done = 0
                xs = starts

                for _ in range(self.longest_label):
                    # generate at most longest_label tokens
                    preds, score, hidden = self.decoder(
                        xs, hidden, enc_out, attn_mask, topk
                    )
                    scores.append(score)
                    xs = preds
                    predictions.append(preds)

                    # check if we've produced the end token
                    for b in range(bsz):
                        if not done[b]:
                            # only add more tokens for examples that aren't done
                            if preds.data[b][0] == self.END_IDX:
                                # if we produced END, we're done
                                done[b] = True
                                total_done += 1
                    if total_done == bsz:
                        # no need to generate any more
                        break

            elif beam_size > 1:
                enc_out, hidden = (
                    encoder_states[0],
                    encoder_states[1],
                )  # take it from encoder
                enc_out = enc_out.unsqueeze(1).repeat(1, beam_size, 1, 1)
                # create batch size num of beams
                data_device = enc_out.device
                beams = [
                    Beam(
                        beam_size,
                        3,
                        0,
                        1,
                        2,
                        min_n_best=beam_size / 2,
                        cuda=data_device,
                    )
                    for _ in range(bsz)
                ]
                # init the input with start token
                xs = starts
                # repeat tensors to support batched beam
                xs = xs.repeat(1, beam_size)
                attn_mask = input_xs.ne(0).float()
                attn_mask = attn_mask.unsqueeze(1).repeat(1, beam_size, 1)
                repeated_hidden = []

                if isinstance(hidden, tuple):
                    for i in range(len(hidden)):
                        repeated_hidden.append(
                            hidden[i].unsqueeze(2).repeat(1, 1, beam_size, 1)
                        )
                    hidden = self.unbeamize_hidden(
                        tuple(repeated_hidden), beam_size, bsz
                    )
                else:  # GRU
                    repeated_hidden = hidden.unsqueeze(2).repeat(1, 1, beam_size, 1)
                    hidden = self.unbeamize_hidden(repeated_hidden, beam_size, bsz)
                enc_out = self.unbeamize_enc_out(enc_out, beam_size, bsz)
                xs = xs.view(bsz * beam_size, -1)
                for step in range(self.longest_label):
                    if all((b.done() for b in beams)):
                        break
                    out = self.decoder(xs, hidden, enc_out)
                    scores = out[1]
                    scores = scores.view(bsz, beam_size, -1)  # -1 is a vocab size
                    for i, b in enumerate(beams):
                        b.advance(F.log_softmax(scores[i, :], dim=-1))
                    xs = torch.cat(
                        [b.get_output_from_current_step() for b in beams]
                    ).unsqueeze(-1)
                    permute_hidden_idx = torch.cat(
                        [
                            beam_size * i + b.get_backtrack_from_current_step()
                            for i, b in enumerate(beams)
                        ]
                    )
                    new_hidden = out[2]
                    if isinstance(hidden, tuple):
                        for i in range(len(hidden)):
                            hidden[i].data.copy_(
                                new_hidden[i].data.index_select(
                                    dim=1, index=permute_hidden_idx
                                )
                            )
                    else:  # GRU
                        hidden.data.copy_(
                            new_hidden.data.index_select(
                                dim=1, index=permute_hidden_idx
                            )
                        )

                for b in beams:
                    b.check_finished()
                beam_pred = [
                    b.get_pretty_hypothesis(b.get_top_hyp()[0])[1:] for b in beams
                ]
                # these beam scores are rescored with length penalty!
                beam_scores = torch.stack([b.get_top_hyp()[1] for b in beams])
                pad_length = max([t.size(0) for t in beam_pred])
                beam_pred = torch.stack(
                    [pad(t, length=pad_length, dim=0) for t in beam_pred], dim=0
                )

                #  prepare n best list for each beam
                n_best_beam_tails = [
                    b.get_rescored_finished(n_best=len(b.finished)) for b in beams
                ]
                nbest_beam_scores = []
                nbest_beam_preds = []
                for i, beamtails in enumerate(n_best_beam_tails):
                    perbeam_preds = []
                    perbeam_scores = []
                    for tail in beamtails:
                        perbeam_preds.append(
                            beams[i].get_pretty_hypothesis(
                                beams[i].get_hyp_from_finished(tail)
                            )
                        )
                        perbeam_scores.append(tail.score)
                    nbest_beam_scores.append(perbeam_scores)
                    nbest_beam_preds.append(perbeam_preds)

                if self.beam_log_freq > 0.0:
                    num_dump = round(bsz * self.beam_log_freq)
                    for i in range(num_dump):
                        dot_graph = beams[i].get_beam_dot(dictionary=self.dict)
                        dot_graph.write_png(
                            os.path.join(
                                self.beam_dump_path,
                                "{}.png".format(self.beam_dump_filecnt),
                            )
                        )
                        self.beam_dump_filecnt += 1

                predictions = beam_pred
                scores = beam_scores

        if isinstance(predictions, list):
            predictions = torch.cat(predictions, 1)
        if isinstance(scores, list):
            scores = torch.cat(scores, 1)

        return (
            predictions,
            scores,
            cand_preds,
            cand_scores,
            encoder_states,
            nbest_beam_preds,
            nbest_beam_scores,
        )


