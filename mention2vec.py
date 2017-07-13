# Author: Karl Stratos (me@karlstratos.com)
import argparse
import dynet as dy
import numpy as np
import random
from collections import Counter
from copy import deepcopy

################################# data #########################################

class Seq(object):
    def __init__(self, w_seq, l_seq=None):
        self.w_seq = w_seq  # word sequence
        self.l_seq = l_seq  # label sequence
        self.bio_pred = []
        self.ent_pred = []
    #TODO: Maybe evaluate here or in SeqData?

class SeqData(object):
    def __init__(self, data_path):
        self.seqs = []
        self.w_enc = {}  # "dog"   -> 35887
        self.c_enc = {}  # "d"     -> 20
        self.l_enc = {}  # "B-ORG" -> 7
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

########################### useful generic operations ##########################

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

################################## model #######################################

class Mention2Vec(object):
    def __init__(self):
        self.__is_training = False
        self.__UNK = "<?>"
        self.__BIO_ENC = {'B': 0, 'I': 1, 'O': 2}
        self.__BIO_DEC = {self.__BIO_ENC[x]: x for x in self.__BIO_ENC}

    def config(self, wdim, cdim, ldim, model_path, wemb_path, epochs,
               dropout_rate):
        self.wdim = wdim
        self.cdim = cdim
        self.ldim = ldim
        self.model_path = model_path
        self.wemb_path = wemb_path
        self.epochs = epochs
        self.dropout_rate = dropout_rate

    def train(self, data, dev=None):
        self.m = dy.ParameterCollection()
        self.__init_params(data)
        self.__set_lstm_dropout()
        self.__is_training = True

        trainer = dy.AdamTrainer(self.m)
        perf_best = 0.
        for epoch in xrange(self.epochs):
            inds = [i for i in xrange(len(data.seqs))]
            random.shuffle(inds)
            for i in inds:
                loss = self.get_loss(data.seqs[i])
                loss.backward()
                trainer.update()
            if dev:
                self.__is_training = False
                perf = self.get_perf(dev)
                if perf > perf_best:
                    perf_best = perf
                    self.save()
                self.__is_training = True

    def get_perf(self, dev):
        for i in xrange(len(dev.seqs)):
            self.get_loss(dev.seqs[i])
        return 0.

    def get_crep(self, w):
        """Character-based representation of word w"""
        inputs = []
        for c in w:
            if self.__is_training and drop(c, self.c_count): c = self.__UNK
            inputs.append(dy.lookup(self.clook, self.c_enc[c]))
        return bilstm_single(inputs, self.clstm1, self.clstm2)

    def get_wemb(self, w):
        """Word embedding of word w"""
        if self.__is_training and drop(w, self.w_count): w = self.__UNK
        return dy.lookup(self.wlook, self.w_enc[w])

    def get_loss_boundary(self, inputs, seq):
        """
        Computes boundary loss for this sequence based on input vectors.
        """
        W_bio1 = dy.parameter(self.l2bio1)
        W_bio1b = dy.parameter(self.l2bio1b)
        W_bio2 = dy.parameter(self.l2bio2)
        W_bio2b = dy.parameter(self.l2bio2b)

        losses = []
        if not self.__is_training: seq.bio_pred = []
        seq.bio_pred = []  # tmp
        for i, h in enumerate(inputs):
            g = W_bio2 * dy.tanh(W_bio1 * h + W_bio1b) + W_bio2b
            if self.__is_training:
                gold = self.__BIO_ENC[seq.l_seq[i][0]]
                losses.append(dy.pickneglogsoftmax(g, gold))
            else:
                seq.bio_pred.append(self.__BIO_DEC[np.argmax(g.npvalue())])
            seq.bio_pred.append(self.__BIO_DEC[np.argmax(g.npvalue())])  #tmp
        boundary_loss = dy.esum(losses)

        return boundary_loss

    def get_boundaries(self, seq):
        """
        Extracts boundaries from sequence. Example for "John Smith was in Paris":
         - If training: {(0,1,"PER"), (4,4,"LOC")}
         - If predicting: {(0,1,None), (4,4,None)}
        """
        bio = [l[0] for l in seq.l_seq] if self.__is_training else seq.bio_pred
        boundaries = []
        i = 0
        while i < len(bio):
            if bio[i] == 'B':
                s = i
                while i < len(bio) and bio[i] != 'O': i += 1
                t = i - 1
                entity = seq.l_seq[s][2:] if self.__is_training else None
                boundaries.append((s, t, entity))
            else:
                i += 1
        return boundaries

    def get_loss_classification(self, inputs, seq):
        """
        Computes classification loss for this sequence based on input vectors.
        """
        W_ent1 = dy.parameter(self.l2ent1)
        W_ent1b = dy.parameter(self.l2ent1b)
        W_ent2 = dy.parameter(self.l2ent2)
        W_ent2b = dy.parameter(self.l2ent2b)

        boundaries = self.get_boundaries(seq)
        losses = []
        if not self.__is_training: seq.ent_pred = []
        seq.ent_pred = []  # tmp
        for (s, t, entity) in boundaries:
            h = bilstm_single(inputs[s:t+1], self.elstm1, self.elstm2)
            g = W_ent2 * dy.tanh(W_ent1 * h + W_ent1b) + W_ent2b
            if self.__is_training:
                gold = self.ent_enc[entity]
                losses.append(dy.pickneglogsoftmax(g, gold))
            else:
                seq.ent_pred.append(self.ent_dec[np.argmax(g.npvalue())])
            seq.ent_pred.append(self.ent_dec[np.argmax(g.npvalue())])  #tmp
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
        print seq.w_seq
        print seq.l_seq
        print seq.bio_pred
        print seq.ent_pred
        print

        return boundary_loss + classification_loss

    def __init_params(self, data):
        self.__init_wparams(data)
        self.__init_cparams(data)
        self.__init_others()

    def __set_lstm_dropout(self):
        self.clstm1.set_dropout(self.dropout_rate)
        self.clstm2.set_dropout(self.dropout_rate)
        self.wlstm1.set_dropout(self.dropout_rate)
        self.wlstm2.set_dropout(self.dropout_rate)
        self.elstm1.set_dropout(self.dropout_rate)
        self.elstm2.set_dropout(self.dropout_rate)

    def __init_wparams(self, data):
        self.w_enc = deepcopy(data.w_enc)
        self.l_enc = deepcopy(data.l_enc)
        self.ent_enc = {}  # "PER" -> 2
        for l in self.l_enc:
            if len(l) > 1 and not l[2:] in self.ent_enc:
                self.ent_enc[l[2:]] = len(self.ent_enc)
        self.ent_dec = {self.ent_enc[x]: x for x in self.ent_enc}
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

        # Entity classification params
        self.elstm1 = dy.LSTMBuilder(1, 2 * self.ldim, 2 * self.ldim, self.m)
        self.elstm2 = dy.LSTMBuilder(1, 2 * self.ldim, 2 * self.ldim, self.m)
        self.l2ent1 = self.m.add_parameters((len(self.ent_enc), 4 * self.ldim))
        self.l2ent1b = self.m.add_parameters((len(self.ent_enc)))
        self.l2ent2 = self.m.add_parameters((len(self.ent_enc),
                                             len(self.ent_enc)))
        self.l2ent2b = self.m.add_parameters((len(self.ent_enc)))

############################  command line usage  ##############################
def main(args):
    data = SeqData(args.data)
    model = Mention2Vec()

    if args.train:
        model.config(args.wdim, args.cdim, args.ldim,
                     args.model, args.emb, args.epochs, args.drop)
        dev = SeqData(args.dev) if args.dev else None
        model.train(data, dev)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", type=str, help="model path")
    argparser.add_argument("data", type=str, help="data for train/test)")
    argparser.add_argument("--dev", type=str, help="data for dev")
    argparser.add_argument("--train", action="store_true", help="train model?")
    argparser.add_argument("--emb", type=str, help="word embeddings")
    argparser.add_argument("--wdim", type=int, default=100)
    argparser.add_argument("--cdim", type=int, default=25)
    argparser.add_argument("--ldim", type=int, default=100)
    argparser.add_argument("--epochs", type=int, default=30)
    argparser.add_argument("--drop", type=float, default=0.1)

    parsed_args = argparser.parse_args()
    main(parsed_args)
