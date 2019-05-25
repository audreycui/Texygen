import numpy as np

from utils.metrics.Metrics import Metrics


class Nll(Metrics):
    def __init__(self, data_loader, rnn, sess):
        super().__init__()
        self.name = 'nll-oracle'
        self.data_loader = data_loader
        self.sess = sess
        self.rnn = rnn

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_score(self):
        return self.nll_loss()

    def nll_loss(self):
        nll = []
        self.data_loader.reset_pointer()
        for it in range(self.data_loader.num_batch):
            sentences, features = self.data_loader.next_batch()
            #print("get nll loss")
            drop_out = .8
            try:
                g_loss = self.rnn.get_nll(self.sess, sentences, features)                                        
            except Exception as e:
                g_loss = self.sess.run(self.rnn.pretrain_loss, 
                                       feed_dict={self.rnn.x: sentences,
                                                  self.rnn.conv_features: np.zeros((self.rnn.batch_size, self.rnn.image_feat_dim), dtype=np.float32),
                                                  self.drop_out: drop_out})
            print("nll loss:", g_loss)  
            nll.append(g_loss)
        return np.mean(nll)
