import sys
import cPickle
from deep_conv import DeepNet
from pyMSSG.pyMSSG import pyMSSG
from conv_net_sentence import make_idx_data_cv


if len(sys.argv) != 2:
    print '{} mssg_path'
    sys.exit(1)

x = cPickle.load(open("mr.p","rb"))
# revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
revs = x[0]

print 'Loading MSSG embeddings...'
m = pyMSSG(sys.argv[1])

datasets = make_idx_data_cv(revs, m.word2id, 0, max_l=56, k=300, filter_h=5)

clf = DeepNet(W.astype("float32"), (64, 300), feature_maps=[300],
              filter_heights=[5], pool_heights=[-1])

clf.fit(datasets[0][:, :-1], datasets[0][:, -1],
        test_set=(datasets[1][:, :-1], datasets[1][:, -1]),
        verbose=True,
        validation=True)

clf.score(datasets[1][:, :-1], datasets[1][:, -1])

