# Author: Karl Stratos (me@karlstratos.com)
"""Mention2Vec implementation"""
import argparse
import dynet as dy
import random
from collections import Counter
from copy import deepcopy

############################# code about data ##################################
class SeqData(object):
    def __init__(self, data_path):
        self.w_seqs, self.l_seqs = [], []
        self.w_enc, self.c_enc, self.l_enc = {}, {}, {}
        self.w_count, self.c_count = Counter(), Counter()
        with open(data_path) as infile:
            w_seq, l_seq = [], []
            for line in infile:
                toks = line.split()
                if toks:
                    w, l = toks[0], toks[1]
                    if not w in self.w_enc: self.w_enc[w] = len(self.w_enc)
                    self.w_count[w] += 1
                    for c in w:
                        if not c in self.c_enc: self.c_enc[c] = len(self.c_enc)
                        self.c_count[c] += 1
                    if not l in self.l_enc: self.l_enc[l] = len(self.l_enc)
                    w_seq.append(w)
                    l_seq.append(l)
                else:
                    if w_seq:
                        self.w_seqs.append(w_seq)
                        self.l_seqs.append(l_seq)
                        w_seq, l_seq = [], []
            if w_seq:
                self.w_seqs.append(w_seq)
                self.l_seqs.append(l_seq)
        #self.w_dec = {self.w_enc[w]: w for w in self.w_enc}
        #self.l_dec = {self.l_enc[l]: l for l in self.l_enc}

############################# code about model #################################
UNK_W, UNK_C = "<w?>", "<c?>"

class Mention2Vec(object):
    def __init__(self):
        self.wdim = 100
        self.cdim = 25
        self.ldim = 100
        self.epochs = 30

    def train(self, data, wemb_path=False):
        self.m = dy.ParameterCollection()
        self.__init_wparams(data, wemb_path)
        self.__init_cparams(data)
        self.__init_lstms()

        for epoch in xrange(self.epochs):
            inds = [i for i in xrange(len(data.w_seqs))]
            random.shuffle(inds)
            for i in inds:
                loss = self.get_loss(data.w_seqs[i], data.l_seqs[i])

    def get_loss(self, w_seq, l_seq):
        dy.renew_cg()

        def dropw(w):
            ct = self.w_count[w]
            return w if random.random() < ct / (ct + 0.25) else UNK_W
        wlook = dy.parameter(self.wlook)

        wreps1 = []
        for w in w_seq:
            crep = self.get_crep(w)
            wind = self.w_enc[dropw(w)]
            wreps1.append(dy.concatenate([crep, dy.lookup(self.wlook, wind)]))

        wreps2 = []
        outs_f, outs_b = [], []
        f, b = self.wlstm1.initial_state(), self.wlstm2.initial_state()
        for wrep_f, wrep_b in zip(wreps1, reversed(wreps1)):
            f, b = f.add_input(wrep_f), b.add_input(wrep_b)
            outs_f.append(f.output())
            outs_b.append(b.output())
        for i, out_b in enumerate(reversed(outs_b)):
            wreps2.append(dy.concatenate([outs_f[i], out_b]))

    def get_crep(self, w):
        def dropc(c):
            ct = self.c_count[c]
            return c if random.random() < ct / (ct + 0.25) else UNK_C

        f, b = self.clstm1.initial_state(), self.clstm2.initial_state()
        for c_f, c_b in zip(w, reversed(w)):
            cind_f = self.c_enc[dropc(c_f)]
            cind_b = self.c_enc[dropc(c_b)]
            f = f.add_input(dy.lookup(self.clook, cind_f))
            b = b.add_input(dy.lookup(self.clook, cind_b))
        return dy.concatenate([f.output(), b.output()])

    def __init_wparams(self, data, wemb_path):
        self.w_enc, self.l_enc = deepcopy(data.w_enc), deepcopy(data.l_enc)
        self.w_count = deepcopy(data.w_count)
        wemb = {}
        if wemb_path:
            with open(wemb_path) as inf:
                for line in inf:
                    toks = line.split()
                    w, emb = toks[0], [float(f) for f in toks[1:]]
                    wemb[w] = emb
                    self.wdim = len(emb)
            for w in wemb:
                if not w in self.w_enc:
                    self.w_enc[w] = len(self.w_enc)
                    self.w_count[w] = 1
        assert not UNK_W in self.w_enc
        self.w_enc[UNK_W] = len(self.w_enc)

        self.wlook = self.m.add_lookup_parameters((len(self.w_enc), self.wdim))
        for w in wemb: self.__wlook.init_row(self.w_enc[w], wemb[w])

    def __init_cparams(self, data):
        self.c_count = deepcopy(data.c_count)
        self.c_enc = {}
        for w in self.w_enc:
            for c in w:
                if not c in self.c_enc:
                    self.c_enc[c] = len(self.c_enc)
                if not c in self.c_count:
                    self.c_count[c] = 1
        assert not UNK_C in self.c_enc
        self.c_enc[UNK_C] = len(self.c_enc)

        self.clook = self.m.add_lookup_parameters((len(self.c_enc), self.cdim))

    def __init_lstms(self):
        self.clstm1 = dy.LSTMBuilder(1, self.cdim, self.cdim, self.m)
        self.clstm2 = dy.LSTMBuilder(1, self.cdim, self.cdim, self.m)
        wlstm_input_dim = self.wdim + 2 * self.cdim
        self.wlstm1 = dy.LSTMBuilder(1, wlstm_input_dim, self.ldim, self.m)
        self.wlstm2 = dy.LSTMBuilder(1, wlstm_input_dim, self.ldim, self.m)

######################## script for command line usage  ########################
def main(args):
    data = SeqData(args.data)
    model = Mention2Vec()
    model.train(data, args.emb)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data", type=str, help="data used for train/test)")
    argparser.add_argument("--emb", type=str, help="word embeddings")

    parsed_args = argparser.parse_args()
    main(parsed_args)
