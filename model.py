from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, EncoderAE, EncoderVAE
import tensorflow as tf

class GCNModelAE(tf.keras.Model):
    def __init__(self, num_features, features_nonzero, dropout=0.0, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.hidden1_dim = 32
        self.hidden2_dim = 16

        self.embeddings = EncoderAE(num_features=self.input_dim,
                                    features_nonzero=self.features_nonzero)

        self.reconstructions = InnerProductDecoder(act=lambda x: x)

    def call(self, inputs):

        x = self.embeddings(inputs)
        x = self.reconstructions(x)

        return x

class GCNModelVAE(tf.keras.Model):
    def __init__(self, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.hidden1 = 32
        self.hidden2 = 16

        self.embeddings = EncoderVAE(num_features=self.input_dim,
                                     num_nodes=self.n_samples,
                                     features_nonzero=self.features_nonzero)

        self.reconstructions = InnerProductDecoder(act=lambda x: x)

    def call(self, inputs):

        x = self.embeddings(inputs)
        x = self.reconstructions(x)

        return x
