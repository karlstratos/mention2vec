# Author: Karl Stratos (me@karlstratos.com)
import argparse
import dynet as dy
import numpy as np
import random
from collections import Counter
from copy import deepcopy

############################# code about data ##################################

class Seq(object):
    def __init__(self, w_seq, l_seq=None):
        self.w_seq = w_seq
        self.l_seq = l_seq
        self.bio_pred = []
        self.ent_pred = []

class SeqData(object):
    def __init__(self, data_path):
        self.seqs = []
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
                        self.seqs.append(Seq(w_seq, l_seq))
                        w_seq, l_seq = [], []
            if w_seq:
                self.seqs.append(Seq(w_seq, l_seq))

############################# code about model #################################
class Mention2Vec(object):
    def __init__(self, wdim, cdim, ldim):
        self.wdim = wdim
        self.cdim = cdim
        self.ldim = ldim
        self.is_training = False

        self.unk = "<?>"
        self.bio_enc = {'B': 0, 'I': 1, 'O': 2}
        self.bio_dec = {self.bio_enc[x]: x for x in self.bio_enc}


    def train(self, data, epochs, model_path, wemb_path=''):
        self.m = dy.ParameterCollection()
        self.__init_wparams(data, wemb_path)
        self.__init_cparams(data)
        self.__init_lstms()

        self.is_training = True
        trainer = dy.AdamTrainer(self.m)

        for epoch in xrange(epochs):
            inds = [i for i in xrange(len(data.seqs))]
            random.shuffle(inds)
            for i in inds:
                loss = self.get_loss(data.seqs[i])
                #print loss.value()
                #print data.seqs[i].bio_pred
                loss.backward()
                trainer.update()

    def drop(self, x, x_count):
        if random.random() < x_count[x] / (x_count[x] + 0.25):
            return x
        else:
            return self.unk

    def get_loss(self, seq):
        dy.renew_cg()
        wlook = dy.parameter(self.wlook)

        wreps1 = []
        for w in seq.w_seq:
            crep = self.get_crep(w)
            if self.is_training: w = self.drop(w, self.w_count)  # Word drop
            wind = self.w_enc[w]
            wreps1.append(dy.concatenate([crep, dy.lookup(self.wlook, wind)]))

        wreps2 = []
        outs_f, outs_b = [], []
        f, b = self.wlstm1.initial_state(), self.wlstm2.initial_state()
        for wrep1_f, wrep1_b in zip(wreps1, reversed(wreps1)):
            f = f.add_input(wrep1_f)
            b = b.add_input(wrep1_b)
            outs_f.append(f.output())
            outs_b.append(b.output())
        for i, out_b in enumerate(reversed(outs_b)):
            wreps2.append(dy.concatenate([outs_f[i], out_b]))

        losses = []
        bio_pred = []
        for i, h in enumerate(wreps2):
            g = self.ff_bio(h)
            if self.is_training:
                gold = self.bio_enc[seq.l_seq[i][0]]
                losses.append(dy.pickneglogsoftmax(g, gold))
            else:
                bio_pred.append(self.bio_dec[np.argmax(g.npvalue())])

            #tmp
            bio_pred.append(self.bio_dec[np.argmax(g.npvalue())])
        #tmp
        seq.bio_pred = bio_pred
        boundary_loss = dy.esum(losses)

        losses = []
        boundaries = self.get_boundaries(seq)
        ent_pred = []
        for (s, t, entity) in boundaries:
            f, b = self.elstm1.initial_state(), self.elstm2.initial_state()
            for wrep2_f, wrep2_b in zip(wreps2[s:t+1], reversed(wreps2[s:t+1])):
                f = f.add_input(wrep2_f)
                b = b.add_input(wrep2_b)
            h = dy.concatenate([f.output(), b.output()])
            g = self.ff_ent(h)
            if self.is_training:
                print entity
                gold = self.ent_enc[entity]
                losses.append(dy.pickneglogsoftmax(g, gold))
            else:
                ent_pred.append(self.ent_dec[np.argmax(g.npvalue())])
            #tmp
            ent_pred.append(self.ent_dec[np.argmax(g.npvalue())])
        #tmp
        seq.ent_pred = ent_pred
        classification_loss = dy.esum(losses) if losses else dy.scalarInput(0.)
        print seq.w_seq
        print seq.l_seq
        print seq.bio_pred
        print seq.ent_pred
        print

        return boundary_loss + classification_loss

    def get_boundaries(self, seq):
        bio = [l[0] for l in seq.l_seq] if self.is_training else seq.bio_pred
        boundaries = []
        i = 0
        while i < len(bio):
            if bio[i] == 'B':
                s = i
                while i < len(bio) and bio[i] != 'O': i += 1
                t = i - 1
                entity = seq.l_seq[s][2:] if self.is_training else None
                boundaries.append((s, t, entity))
            else:
                i += 1
        return boundaries

    def ff_ent(self, h):
        W_ent1 = dy.parameter(self.l2ent1)
        W_ent1b = dy.parameter(self.l2ent1b)
        W_ent2 = dy.parameter(self.l2ent2)
        W_ent2b = dy.parameter(self.l2ent2b)
        return W_ent2 * dy.tanh(W_ent1 * h + W_ent1b) + W_ent2b

    def ff_bio(self, h):
        W_bio1 = dy.parameter(self.l2bio1)
        W_bio1b = dy.parameter(self.l2bio1b)
        W_bio2 = dy.parameter(self.l2bio2)
        W_bio2b = dy.parameter(self.l2bio2b)
        return W_bio2 * dy.tanh(W_bio1 * h + W_bio1b) + W_bio2b

    def get_crep(self, w):
        if self.is_training:
            w_tmp = [self.drop(c, self.c_count) for c in w]  # Char drop
            w = w_tmp
        f, b = self.clstm1.initial_state(), self.clstm2.initial_state()
        for c_f, c_b in zip(w, reversed(w)):
            cind_f = self.c_enc[c_f]
            cind_b = self.c_enc[c_b]
            f = f.add_input(dy.lookup(self.clook, cind_f))
            b = b.add_input(dy.lookup(self.clook, cind_b))
        return dy.concatenate([f.output(), b.output()])

    def __init_wparams(self, data, wemb_path):
        self.w_enc = deepcopy(data.w_enc)
        self.l_enc = deepcopy(data.l_enc)
        self.ent_enc = {}
        for l in self.l_enc:
            if len(l) > 1 and not l[2:] in self.ent_enc:
                self.ent_enc[l[2:]] = len(self.ent_enc)
        self.ent_dec = {self.ent_enc[x]: x for x in self.ent_enc}
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
        assert not self.unk in self.w_enc
        self.w_enc[self.unk] = len(self.w_enc)


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
        assert not self.unk in self.c_enc
        self.c_enc[self.unk] = len(self.c_enc)

        self.clook = self.m.add_lookup_parameters((len(self.c_enc), self.cdim))

    def __init_lstms(self):
        self.clstm1 = dy.LSTMBuilder(1, self.cdim, self.cdim, self.m)
        self.clstm2 = dy.LSTMBuilder(1, self.cdim, self.cdim, self.m)

        wc_dim = self.wdim + 2 * self.cdim
        self.wlstm1 = dy.LSTMBuilder(1, wc_dim, self.ldim, self.m)
        self.wlstm2 = dy.LSTMBuilder(1, wc_dim, self.ldim, self.m)

        self.l2bio1 = self.m.add_parameters((3, 2 * self.ldim))
        self.l2bio1b = self.m.add_parameters((3))
        self.l2bio2 = self.m.add_parameters((3, 3))
        self.l2bio2b = self.m.add_parameters((3))

        self.elstm1 = dy.LSTMBuilder(1, 2 * self.ldim, 2 * self.ldim, self.m)
        self.elstm2 = dy.LSTMBuilder(1, 2 * self.ldim, 2 * self.ldim, self.m)

        self.l2ent1 = self.m.add_parameters((len(self.ent_enc), 4 * self.ldim))
        self.l2ent1b = self.m.add_parameters((len(self.ent_enc)))
        self.l2ent2 = self.m.add_parameters((len(self.ent_enc),
                                             len(self.ent_enc)))
        self.l2ent2b = self.m.add_parameters((len(self.ent_enc)))


######################## script for command line usage  ########################
def main(args):
    data = SeqData(args.data)
    model = Mention2Vec(args.wdim, args.cdim, args.ldim)

    if args.train:
        model.train(data, args.epochs, args.model, args.emb)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("model", type=str, help="model path")
    argparser.add_argument("data", type=str, help="data used for train/test)")
    argparser.add_argument("--train", action="store_true", help="train model?")
    argparser.add_argument("--emb", type=str, help="word embeddings")
    argparser.add_argument("--wdim", type=int, default=100)
    argparser.add_argument("--cdim", type=int, default=25)
    argparser.add_argument("--ldim", type=int, default=100)
    argparser.add_argument("--epochs", type=int, default=30)

    parsed_args = argparser.parse_args()
    main(parsed_args)
