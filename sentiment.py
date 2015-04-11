import cPickle
from deep_conv import DeepNet
from conv_net_sentence import make_idx_data_cv

x = cPickle.load(open("mr.p","rb"))
revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]

datasets = make_idx_data_cv(revs, word_idx_map, 0, max_l=56, k=300, filter_h=5)

clf = DeepNet(W.astype("float32"), (64, 300), feature_maps=[300],
              filter_heights=[5], pool_heights=[-1])

clf.fit(datasets[0][:, :-1], datasets[0][:, -1],
        test_set=(datasets[1][:, :-1], datasets[1][:, -1]),
        verbose=True,
        validation=True)

clf.score(datasets[1][:, :-1], datasets[1][:, -1])

