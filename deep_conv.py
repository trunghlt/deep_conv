from conv_net_classes import *
from conv_net_sentence import *
from abc import ABCMeta
from sklearn.externals import six
from sklearn.base import BaseEstimator, ClassifierMixin
from theano.tensor.signal import downsample
import theano.tensor as T

class DeepNet(six.with_metaclass(ABCMeta, BaseEstimator), ClassifierMixin):
    
    def __init__(self, U, img_size, dropout_rate=[0.5], shuffle_batch=True,
            n_epochs=25, batch_size=50, lr_decay = 0.95, conv_non_linear="relu",
            feature_maps=[300, 100],
            filter_heights=[5, 5],
            pool_heights=[1, 1],
            activations=[Iden],
            sqr_norm_lim=9, 
            mlp_layers=[2],
            non_static=True):        
        assert len(feature_maps)==len(filter_heights), "feature_maps and filter_heights must have the same size."
        assert len(feature_maps)==len(pool_heights), "feature_maps and pool_heights must have the same size."
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle_batch = shuffle_batch
        
        rng = np.random.RandomState(3435)
        img_h, img_w = img_size

        # Define model architecture
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        Words = theano.shared(value=U, name="Words", borrow=False)
        params = []
        
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(img_w, dtype="float32")
        self._set_zero = theano.function(
                [zero_vec_tensor], 
                updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))]
            )
        
        inp = Words[T.cast(self.x.flatten(), dtype="int32")]\
                .reshape((self.x.shape[0], 1, self.x.shape[1], Words.shape[1]))
        inp_h, inp_w = img_h, img_w
        conv_layers = []
        all_pooled = []
        all_fm = 0
        for fm, filter_h, pool_h in zip(feature_maps, filter_heights, pool_heights):
            if pool_h == -1:
                pool_h = inp_h - filter_h + 1
            layer = LeNetConvPoolLayer(
                        rng,
                        input=inp,
                        image_shape=(batch_size, 1, inp_h, inp_w),
                        filter_shape=(fm, 1, filter_h, inp_w),
                        poolsize=(pool_h, 1),
                        non_linear=conv_non_linear
                    )
            all_pooled.append(downsample.max_pool_2d(input=layer.output, ds=(inp_h-filter_h+1, 1), ignore_border=True))
            conv_layers.append(layer)
            all_fm += fm
            
            params += layer.params           
            inp_h, inp_w = (img_h - filter_h + 1)/pool_h, fm
            # convert [batch_size, fm, H, 1] => [batch_size, 1, H, fm]
            inp = theano.tensor.transpose(layer.output, axes=(0, 3, 2, 1))    

        # last mlp layer
        classifier = MLPDropout(
                        rng, 
                        input=inp.flatten(2), 
                        layer_sizes=[all_fm] + mlp_layers, 
                        activations=activations, 
                        dropout_rates=dropout_rate
                     )
        params += classifier.params

        if non_static:
            params += [Words]
            
        self.conv_layers = conv_layers
        self.classifier = classifier
        self.cost = classifier.negative_log_likelihood(self.y)
        self.error = classifier.errors(self.y)
        dropout_cost = classifier.dropout_negative_log_likelihood(self.y)
        self.grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)            
        self.y_pred = classifier.y_pred
        
    def _prepare(self, X, Y):
        #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
        #extra data (at random)
        assert X.shape[0]==Y.shape[0]
        np.random.seed(3435)
        if X.shape[0] % self.batch_size > 0:
            extra_data_num = self.batch_size - X.shape[0] % self.batch_size
            idx = np.random.permutation(range(X.shape[0]))
            X = np.append(X, X[idx[:extra_data_num]], axis=0)
            Y = np.append(Y, Y[idx[:extra_data_num]], axis=0)
        idx = np.random.permutation(range(X.shape[0]))
        return X[idx], Y[idx]
    
    def fit(self, X, Y, verbose=False, validation=False, test_set=None):
        X, Y = self._prepare(X, Y)
        n_batches = X.shape[0]/self.batch_size
        X, Y = shared_dataset((X, Y), borrow=True)
        index = T.lscalar()
        train_model = theano.function(
            [index], 
            self.cost, 
            updates=self.grad_updates,
            givens={
                self.x: X[index*self.batch_size:(index + 1)*self.batch_size],
                self.y: Y[index*self.batch_size:(index + 1)*self.batch_size]
            })        

        test_model = theano.function(
            [index], 
            self.error, 
            givens={
                self.x: X[index*self.batch_size:(index + 1)*self.batch_size],
                self.y: Y[index*self.batch_size:(index + 1)*self.batch_size]
            })        

        for epoch in xrange(1, self.n_epochs + 1):
            if verbose:
                print "Epoch %d..." % epoch
            
            n_train_batches = int(round(.9*n_batches)) if validation else n_batches
            idx = np.random.permutation(range(n_batches)) if self.shuffle_batch else range(n_batches)
            
            train_idx = idx[:n_train_batches] if validation else idx
            for minibatch_index in train_idx:
                c = train_model(minibatch_index)
                self._set_zero(self.zero_vec)

            if validation:
                val_error = np.mean([test_model(i) for i in idx[n_train_batches:]])
                
            if verbose:
                if validation:
                    print "  --> Val score: %.2f%%" % ((1 - val_error)*100)
                if test_set is not None:
                    print "  --> Test score: %.2f%%" % (self.score(*test_set)*100)
                    
    def predict(self, X):
        size = X.shape[0]
        X = np.append(X, X[:(self.batch_size - 1) - (size - 1) % self.batch_size], axis=0)
        n_batches = X.shape[0]/self.batch_size
        X, _Y = shared_dataset((X, []), borrow=True)        
        y = []
        index = T.lscalar()
        y_pred = theano.function(
            [index], 
            self.y_pred, 
            givens={
                self.x: X[index*self.batch_size:(index + 1)*self.batch_size]
            })            
        for i in xrange(n_batches):
            y = np.concatenate((y, y_pred(i)))
        return np.asarray(y[:size])
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return (y==y_pred).mean()
