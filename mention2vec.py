# Author: Karl Stratos (me@karlstratos.com)
import argparse
import dynet as dy
import numpy as np
import os
import pickle
import random
import sys
import time
from collections import Counter
from copy import deepcopy

########################### useful generic operations ##########################

def get_boundaries(bio):
    """
    Extracts an ordered list of boundaries. BIO label sequences can be either
    -     Raw BIO: B     I     I     O => {(0, 2, None)}
    - Labeled BIO: B-PER I-PER B-LOC O => {(0, 1, "PER"), (2, 2, "LOC")}
    """
    boundaries= []
    i = 0

    while i < len(bio):
        if bio[i][0] == 'O': i += 1
        else:
            s = i
            entity = bio[s][2:] if len(bio[s]) > 2 else None
            i += 1
            while i < len(bio) and bio[i][0] == 'I':
                if len(bio[i]) > 2 and bio[i][2:] != entity: break
                i += 1
            boundaries.append((s, i - 1, entity))

    return boundaries

def label_bio(bio, ents):
    labeled_bio = []
    i = 0
    counter = 0
    while i < len(bio):
        if bio[i][0] == 'O':
            labeled_bio.append('O')
            i += 1
        else:
            labeled_bio.append(bio[i][0] + '-' + ents[counter])
            i += 1
            while i < len(bio) and bio[i][0] == 'I':
                labeled_bio.append(bio[i][0] + '-' + ents[counter])
                i += 1
            counter += 1

    return labeled_bio

def score_crf(start_b, T, end_b, score_vecs, inds):
    total = start_b[inds[0]] + score_vecs[0][inds[0]]
    for i in xrange(1, len(score_vecs)):
        total += T[inds[i-1]][inds[i]] + score_vecs[i][inds[i]]
    total += end_b[inds[-1]]
    return total

def viterbi(start_b, T, end_b, score_vecs, valid):
    num_labels = len(valid)
    pi = [[None] * num_labels] * len(score_vecs)
    bp = [[None] * num_labels] * len(score_vecs)

    for y in xrange(num_labels): pi[0][y] = score_vecs[0][y] + start_b[y]

    for i in xrange(1, len(score_vecs)):
        for y in xrange(num_labels):
            score_best = float("-inf")
            y_prev_best = None
            valid_previous_labels = valid[y]
            for y_prev in valid_previous_labels:
                score = pi[i-1][y_prev] +  T[y_prev][y] + score_vecs[i][y]
                if score > score_best:
                    y_prev_best = y_prev
                    score_best = score
            pi[i][y] = score_best
            bp[i][y] = y_prev_best

    best_y = np.argmax([pi[-1][y] + end_b[y] for y in xrange(num_labels)])
    pred_rev = [best_y]
    for i in reversed(xrange(1, len(score_vecs))):
        best_y = bp[i][best_y]
        pred_rev.append(best_y)

    return pred_rev[::-1]

def drop(x, x_count):
    """Drops x with higher probabiliy if x is less frequent."""
    return random.random() > x_count[x] / (x_count[x] + 0.25)

def bilstm_single(inputs, lstm1, lstm2):
    """Computes a single embedding of input expressions using 2 LTSMs."""
    f = lstm1.initial_state()
    b = lstm2.initial_state()
    for input_f, input_b in zip(inputs, reversed(inputs)):
        f = f.add_input(input_f)
        b = b.add_input(input_b)
    return dy.concatenate([f.output(), b.output()])

def bilstm(inputs, lstm1, lstm2):
    """Computes embeddings of input expressions using 2 LTSMs."""
    f = lstm1.initial_state()
    b = lstm2.initial_state()
    outs_f = []
    outs_b = []
    for input_f, input_b in zip(inputs, reversed(inputs)):
        f = f.add_input(input_f)
        b = b.add_input(input_b)
        outs_f.append(f.output())
        outs_b.append(b.output())

    outs = []
    for i, out_b in enumerate(reversed(outs_b)):
        outs.append(dy.concatenate([outs_f[i], out_b]))
    return outs

################################# data #########################################

class Seq(object):
    def __init__(self, w_seq, l_seq=None):
        self.w_seq = w_seq  # word sequence
        self.l_seq = l_seq  # label sequence
        self.bio_pred = []
        self.ent_pred = []

    def evaluate(self, tp, fp, fn, all_ent="<all>"):
        gold_boundaries = get_boundaries(self.l_seq)
        pred_boundaries_untyped = get_boundaries(self.bio_pred)
        pred_boundaries = []
        for i in xrange(len(pred_boundaries_untyped)):
            s, t, _ = pred_boundaries_untyped[i]
            entity = self.ent_pred[i]
            pred_boundaries.append((s, t, entity))
        gold_boundaries = set(gold_boundaries)
        pred_boundaries = set(pred_boundaries)
        for (s, t, entity) in gold_boundaries:
            if (s, t, entity) in pred_boundaries:
                tp[entity] += 1
                tp[all_ent] += 1
            else:
                fn[entity] += 1
                fn[all_ent] += 1
        for (s, t, entity) in pred_boundaries:
            if not (s, t, entity) in gold_boundaries:
                fp[entity] += 1
                fp[all_ent] += 1

class SeqData(object):
    def __init__(self, data_path):
        self.seqs = []
        self.w_enc = {}  # "dog"   -> 35887
        self.c_enc = {}  # "d"     -> 20
        self.l_enc = {}  # "B-ORG" -> 7
        self.e_enc = {}  # "PER" -> 2
        self.__ALL = "<all>"  # Denotes all entity types.
        self.w_count = Counter()
        self.c_count = Counter()

        with open(data_path) as infile:
            w_seq = []
            l_seq = []
            for line in infile:
                toks = line.split()
                if toks:
                    w = toks[0]
                    l = toks[1]
                    self.w_count[w] += 1
                    if not w in self.w_enc: self.w_enc[w] = len(self.w_enc)
                    for c in w:
                        self.c_count[c] += 1
                        if not c in self.c_enc: self.c_enc[c] = len(self.c_enc)
                    if not l in self.l_enc: self.l_enc[l] = len(self.l_enc)
                    w_seq.append(w)
                    l_seq.append(l)
                else:
                    if w_seq:
                        self.seqs.append(Seq(w_seq, l_seq))
                        w_seq = []
                        l_seq = []
            if w_seq:
                self.seqs.append(Seq(w_seq, l_seq))

        for l in self.l_enc:
            if len(l) > 1 and not l[2:] in self.e_enc:
                self.e_enc[l[2:]] = len(self.e_enc)

    def evaluate(self):
        keys = self.e_enc.keys() + [self.__ALL]
        tp = {e: 0 for e in keys}
        fp = {e: 0 for e in keys}
        fn = {e: 0 for e in keys}
        for seq in self.seqs:
            seq.evaluate(tp, fp, fn, self.__ALL)

        self.p = {}
        self.r = {}
        for e in keys:
            pZ = tp[e] + fp[e]
            rZ = tp[e] + fn[e]
            self.p[e] = 100. * tp[e] / pZ if pZ > 0. else 0.
            self.r[e] = 100. * tp[e] / rZ if rZ > 0. else 0.
        return self.f1(self.__ALL)

    def f1(self, cat):
        f1Z = self.p[cat] + self.r[cat]
        f1 = 2. * self.p[cat] * self.r[cat] / f1Z if f1Z > 0. else 0.0
        return f1

    def write(self, path):
        with open(path, 'w') as outf:
            for seq in self.seqs:

                pred = label_bio(seq.bio_pred, seq.ent_pred)
                for i in xrange(len(seq.w_seq)):
                    outf.write(seq.w_seq[i] + " " + seq.l_seq[i] + " " + pred[i]
                               + "\n")
                outf.write("\n")

################################## model #######################################

class Mention2Vec(object):
    def __init__(self):
        self.__is_training = False
        self.__UNK = "<?>"
        self.__BIO_ENC = {'B': 0, 'I': 1, 'O': 2}
        self.__BIO_DEC = {self.__BIO_ENC[x]: x for x in self.__BIO_ENC}

    def config(self, wdim, cdim, ldim, model_path, wemb_path, epochs,
               loss, dropout_rate, learning_rate):
        self.wdim = wdim
        self.cdim = cdim
        self.ldim = ldim
        self.model_path = model_path
        self.wemb_path = wemb_path
        self.epochs = epochs
        self.loss = loss
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def train(self, data, dev=None):
        self.m = dy.ParameterCollection()
        self.__init_params(data)
        self.__enable_lstm_dropout()
        self.__is_training = True
        if os.path.isfile(self.model_path): os.remove(self.model_path)
        if not os.path.exists(self.model_path): os.makedirs(self.model_path)

        trainer = dy.AdamTrainer(self.m, self.learning_rate)
        perf_best = 0.
        exists = False
        for epoch in xrange(self.epochs):
            inds = [i for i in xrange(len(data.seqs))]
            random.shuffle(inds)
            for i in inds:
                loss = self.get_loss(data.seqs[i])
                loss.backward()
                trainer.update()

            if dev:
                self.__is_training = False
                self.__disable_lstm_dropout()
                perf, _ = self.get_perf(dev)
                print "Epoch {0:d} F1: {1:.2f}".format(epoch + 1, perf),
                if perf > perf_best:
                    perf_best = perf
                    print 'new best - saving model',
                    self.save()
                    exists = True
                self.__is_training = True
                self.__enable_lstm_dropout()
                print

            else:
                self.save()

        if exists:
            m = Mention2Vec()
            m.load_and_populate(self.model_path)
            perf, _ = m.get_perf(dev)
            print "Best dev F1: {0:.2f}".format(perf)

    def save(self):
        self.m.save(os.path.join(self.model_path, "model"))
        with open(os.path.join(self.model_path, "info.pickle"), 'w') as outf:
            pickle.dump((self.w_enc, self.wdim, self.c_enc, self.cdim,
                         self.ldim, self.e_dec, self.loss), outf)

    def load_and_populate(self, model_path):
        self.m = dy.ParameterCollection()
        self.model_path = model_path
        with open(os.path.join(self.model_path, "info.pickle")) as inf:
            self.w_enc, self.wdim, self.c_enc, self.cdim, self.ldim, \
                self.e_dec, self.loss = pickle.load(inf)
        self.wlook = self.m.add_lookup_parameters((len(self.w_enc), self.wdim))
        self.clook = self.m.add_lookup_parameters((len(self.c_enc), self.cdim))
        self.__init_others()
        self.m.populate(os.path.join(self.model_path, "model"))
        self.__disable_lstm_dropout()

    def get_perf(self, test):
        start_time = time.time()
        num_words = 0
        for i in xrange(len(test.seqs)):
            self.get_loss(test.seqs[i])
            num_words += len(test.seqs[i].w_seq)
        return test.evaluate(), int(num_words / (time.time() - start_time))

    def get_crep(self, w):
        """Character-based representation of word w"""
        inputs = []
        for c in w:
            if self.__is_training and drop(c, self.c_count): c = self.__UNK
            if not c in self.c_enc: c = self.__UNK
            inputs.append(dy.lookup(self.clook, self.c_enc[c]))
        return bilstm_single(inputs, self.clstm1, self.clstm2)

    def get_wemb(self, w):
        """Word embedding of word w"""
        if self.__is_training and drop(w, self.w_count): w = self.__UNK
        if not w in self.w_enc: w = self.__UNK
        return dy.lookup(self.wlook, self.w_enc[w])

    def get_loss_boundary(self, inputs, seq):
        """
        Computes boundary loss for this sequence based on input vectors.
        """
        W_bio1 = dy.parameter(self.l2bio1)
        W_bio1b = dy.parameter(self.l2bio1b)
        W_bio2 = dy.parameter(self.l2bio2)
        W_bio2b = dy.parameter(self.l2bio2b)
        def ff(h): return W_bio2 * dy.tanh(W_bio1 * h + W_bio1b) + W_bio2b

        gs = [ff(h) for h in inputs]  # Inputs now 3 dimensional ("BIO scores")

        if self.loss == "global":
            boundary_loss = self.get_loss_boundary_global(gs, seq)
        elif self.loss == "local":
            boundary_loss = self.get_loss_boundary_local(gs, seq)
        else:
            sys.exit("Unknown loss \"{0}\"".format(self.loss))

        losses = []
        if not self.__is_training: seq.bio_pred = []
        for i, g in enumerate(gs):
            if self.__is_training:
                gold = self.__BIO_ENC[seq.l_seq[i][0]]
                losses.append(dy.pickneglogsoftmax(g, gold))
            else:
                seq.bio_pred.append(self.__BIO_DEC[np.argmax(g.npvalue())])
        boundary_loss = dy.esum(losses) if losses else dy.scalarInput(0.)

        return boundary_loss

    def get_loss_boundary_global(self, score_vecs, seq):
        start_b = dy.parameter(self.start_bias)
        T = dy.parameter(self.trans_mat)
        end_b = dy.parameter(self.end_bias)

        if not self.__is_training:
            seq.bio_pred = viterbi(start_b, T, end_b, score_vecs, self.valid)
            return dy.scalarInput(0.)

        pi = [[None] * 3] * len(score_vecs)

        for y in xrange(3): pi[0][y] = score_vecs[0][y] + start_b[y]

        for i in xrange(1, len(pi)):
            for y in xrange(3):
                pi[i][y] = dy.logsumexp([pi[i-1][y_prev] +
                                         T[y_prev][y] + score_vecs[i][y]
                                         for y_prev in xrange(3)])

        normalizer = dy.logsumexp([pi[-1][y] + end_b[y] for y in xrange(3)])
        gold_score = score_crf(start_b, T, end_b, score_vecs,
                               [self.__BIO_ENC[l[0]] for l in seq.l_seq])

        return normalizer - gold_score

    def get_loss_boundary_local(self, score_vecs, seq):
        losses = []
        if not self.__is_training: seq.bio_pred = []
        for i, scores in enumerate(score_vecs):
            if self.__is_training:
                gold = self.__BIO_ENC[seq.l_seq[i][0]]
                losses.append(dy.pickneglogsoftmax(scores, gold))
            else:
                scores_np = scores.npvalue()
                if i > 0:
                    y_best = None
                    score_best = float("-inf")
                    for y in xrange(3):
                        if self.__BIO_ENC[seq.bio_pred[i-1]] in self.valid[y] \
                           and scores_np[y] > score_best:
                            y_best = y
                            score_best = scores_np[y]
                else:
                    y_best = np.argmax(scores_np)

                seq.bio_pred.append(self.__BIO_DEC[y_best])

        return dy.esum(losses) if losses else dy.scalarInput(0.)

    def get_loss_classification(self, inputs, seq):
        """
        Computes classification loss for this sequence based on input vectors.
        """
        W_ent1 = dy.parameter(self.l2ent1)
        W_ent1b = dy.parameter(self.l2ent1b)
        W_ent2 = dy.parameter(self.l2ent2)
        W_ent2b = dy.parameter(self.l2ent2b)
        def ff(h): return W_ent2 * dy.tanh(W_ent1 * h + W_ent1b) + W_ent2b

        boundaries = get_boundaries(seq.l_seq) if self.__is_training else \
                     get_boundaries(seq.bio_pred)
        losses = []
        if not self.__is_training: seq.ent_pred = []
        for (s, t, entity) in boundaries:
            h = bilstm_single(inputs[s:t+1], self.elstm1, self.elstm2)
            g = ff(h)
            if self.__is_training:
                gold = self.e_enc[entity]
                losses.append(dy.pickneglogsoftmax(g, gold))
            else:
                seq.ent_pred.append(self.e_dec[np.argmax(g.npvalue())])
        classification_loss = dy.esum(losses) if losses else dy.scalarInput(0.)

        return classification_loss

    def get_loss(self, seq):
        """Compute loss for this sequence & fill in predictions."""
        dy.renew_cg()

        wreps1 = [dy.concatenate([self.get_crep(w),
                                  self.get_wemb(w)]) for w in seq.w_seq]
        wreps2 = bilstm(wreps1, self.wlstm1, self.wlstm2)

        boundary_loss = self.get_loss_boundary(wreps2, seq)
        classification_loss = self.get_loss_classification(wreps2, seq)

        return boundary_loss + classification_loss

    def __init_params(self, data):
        self.__init_wparams(data)
        self.__init_cparams(data)
        self.__init_others()

    def __enable_lstm_dropout(self):
        self.clstm1.set_dropout(self.dropout_rate)
        self.clstm2.set_dropout(self.dropout_rate)
        self.wlstm1.set_dropout(self.dropout_rate)
        self.wlstm2.set_dropout(self.dropout_rate)
        self.elstm1.set_dropout(self.dropout_rate)
        self.elstm2.set_dropout(self.dropout_rate)

    def __disable_lstm_dropout(self):
        self.clstm1.disable_dropout()
        self.clstm2.disable_dropout()
        self.wlstm1.disable_dropout()
        self.wlstm2.disable_dropout()
        self.elstm1.disable_dropout()
        self.elstm2.disable_dropout()

    def __init_wparams(self, data):
        self.w_enc = deepcopy(data.w_enc)
        self.l_enc = deepcopy(data.l_enc)
        self.e_enc = deepcopy(data.e_enc)
        self.e_dec = {self.e_enc[x]: x for x in self.e_enc}
        self.w_count = deepcopy(data.w_count)
        wemb = {}
        if self.wemb_path:
            with open(self.wemb_path) as inf:
                for line in inf:
                    toks = line.split()
                    w, emb = toks[0], [float(f) for f in toks[1:]]
                    wemb[w] = emb
                    self.wdim = len(emb)  # Override word dimension.
            for w in wemb:
                if not w in self.w_enc:
                    self.w_enc[w] = len(self.w_enc)
                    self.w_count[w] = 1
        assert not self.__UNK in self.w_enc
        self.w_enc[self.__UNK] = len(self.w_enc)

        self.wlook = self.m.add_lookup_parameters((len(self.w_enc), self.wdim))
        for w in wemb:
            self.wlook.init_row(self.w_enc[w], wemb[w])

    def __init_cparams(self, data):
        self.c_enc = deepcopy(data.c_enc)
        self.c_count = deepcopy(data.c_count)
        for w in self.w_enc:
            for c in w:
                if not c in self.c_enc:
                    self.c_enc[c] = len(self.c_enc)
                    self.c_count[c] = 1
        assert not self.__UNK in self.c_enc
        self.c_enc[self.__UNK] = len(self.c_enc)

        self.clook = self.m.add_lookup_parameters((len(self.c_enc), self.cdim))

    def __init_others(self):
        # Representation LSTM builders
        self.clstm1 = dy.LSTMBuilder(1, self.cdim, self.cdim, self.m)
        self.clstm2 = dy.LSTMBuilder(1, self.cdim, self.cdim, self.m)
        self.wlstm1 = dy.LSTMBuilder(1, self.wdim + 2 * self.cdim, self.ldim,
                                     self.m)
        self.wlstm2 = dy.LSTMBuilder(1, self.wdim + 2 * self.cdim, self.ldim,
                                     self.m)

        # BIO sequence params
        self.l2bio1 = self.m.add_parameters((3, 2 * self.ldim))
        self.l2bio1b = self.m.add_parameters((3))
        self.l2bio2 = self.m.add_parameters((3, 3))
        self.l2bio2b = self.m.add_parameters((3))
        if self.loss == "global":
            self.start_bias = self.m.add_parameters((3))
            self.trans_mat = self.m.add_parameters((3, 3))
            self.end_bias = self.m.add_parameters((3))

        # valid[y] = set of labels that can precede y
        self.valid = {self.__BIO_ENC['B']: [self.__BIO_ENC['B'],
                                            self.__BIO_ENC['I'],
                                            self.__BIO_ENC['O']],

                      self.__BIO_ENC['I']: [self.__BIO_ENC['B'],
                                            self.__BIO_ENC['I']],

                      self.__BIO_ENC['O']: [self.__BIO_ENC['B'],
                                            self.__BIO_ENC['I'],
                                            self.__BIO_ENC['O']]}

        # Entity classification params
        self.elstm1 = dy.LSTMBuilder(1, 2 * self.ldim, 2 * self.ldim, self.m)
        self.elstm2 = dy.LSTMBuilder(1, 2 * self.ldim, 2 * self.ldim, self.m)
        self.l2ent1 = self.m.add_parameters((len(self.e_dec), 4 * self.ldim))
        self.l2ent1b = self.m.add_parameters((len(self.e_dec)))
        self.l2ent2 = self.m.add_parameters((len(self.e_dec), len(self.e_dec)))
        self.l2ent2b = self.m.add_parameters((len(self.e_dec)))

############################  command line usage  ##############################

def main(args):
    random.seed(42)
    data = SeqData(args.data)
    model = Mention2Vec()

    if args.train:
        model.config(args.wdim, args.cdim, args.ldim,
                     args.model, args.emb, args.epochs, args.loss,
                     args.drop, args.lrate)
        dev = SeqData(args.dev) if args.dev else None
        model.train(data, dev)

    else:
        model.load_and_populate(args.model)
        perf, speed = model.get_perf(data)
        if args.pred: data.write(args.pred)
        print "F1: {0:.2f} ({1} words/sec)".format(perf, speed)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", type=str, help="model path")
    argparser.add_argument("data", type=str, help="data for train/test)")
    argparser.add_argument("--dev", type=str, help="data for dev")
    argparser.add_argument("--pred", type=str, help="write predictions here")
    argparser.add_argument("--train", action="store_true", help="train model?")
    argparser.add_argument("--emb", type=str, help="word embeddings")
    argparser.add_argument("--wdim", type=int, default=100, help="%(default)d")
    argparser.add_argument("--cdim", type=int, default=25, help="%(default)d")
    argparser.add_argument("--ldim", type=int, default=100, help="%(default)d")
    argparser.add_argument("--epochs", type=int, default=30,
                           help="%(default)d")
    argparser.add_argument("--loss", type=str, default="global",
                           help="%(default)s")
    argparser.add_argument("--drop", type=float, default=0.4,
                           help="%(default)f")
    argparser.add_argument("--lrate", type=float, default=0.0005,
                           help="%(default)f")
    argparser.add_argument("--dynet-mem")
    argparser.add_argument("--dynet-seed")

    parsed_args = argparser.parse_args()
    main(parsed_args)
