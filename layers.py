import tensorflow as tf

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.math.floor(random_tensor), dtype=tf.bool)
#     print(x)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class GraphConvolution(tf.keras.layers.Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        
        self.w = self.add_weight(
            shape=(input_dim, output_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act

    def call(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        x = tf.keras.layers.Dropout(1-self.dropout)(x)
        x = tf.matmul(x, self.w)
        x = tf.sparse.sparse_dense_matmul(adj, x)
        # x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(tf.keras.layers.Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        
        self.w = self.add_weight(
            shape=(input_dim, output_dim),
            initializer="random_normal",
            trainable=True,
        )
        
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def call(self, inputs):
        x = inputs[0]
        adj = inputs[1]
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse.sparse_dense_matmul(x, self.w)
        x = tf.sparse.sparse_dense_matmul(adj, x)
        outputs = self.act(x)
        return outputs

class EncoderAE(tf.keras.layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, num_features, features_nonzero, dropout=0.0, **kwargs):
        super(EncoderAE, self).__init__(**kwargs)

        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.dropout = dropout
        self.hidden1_dim = 32
        self.hidden2_dim = 16
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout)

        self.embeddings = GraphConvolution(input_dim=self.hidden1_dim,
                                           output_dim=self.hidden2_dim,
                                           act=lambda x: x,
                                           dropout=self.dropout)

    def call(self, inputs):
        x = self.hidden1(inputs)
        z_mean = self.embeddings([x, inputs[1]]) 
        return z_mean


class EncoderVAE(tf.keras.layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, num_features, num_nodes, features_nonzero, dropout=0.0, **kwargs):
        super(EncoderVAE, self).__init__(**kwargs)

        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.dropout = dropout
        self.hidden1_dim = 32
        self.hidden2_dim = 16

        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout)

        self.z_mean = GraphConvolution(input_dim=self.hidden1_dim,
                                           output_dim=self.hidden2_dim,
                                           act=lambda x: x,
                                           dropout=self.dropout)

        self.z_log_std = GraphConvolution(input_dim=self.hidden1_dim,
                                          output_dim=self.hidden2_dim,
                                          act=lambda x: x,
                                          dropout=self.dropout)

    def call(self, inputs):

        x = self.hidden1(inputs)
        z_mean = self.z_mean([x, inputs[1]])
        z_log_std = self.z_log_std([x, inputs[1]])
        self.z = z_mean + tf.random.normal([self.n_samples, self.hidden2_dim]) * tf.math.exp(z_log_std)
        return self.z


class InnerProductDecoder(tf.keras.layers.Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def call(self, inputs):
        inputs = tf.keras.layers.Dropout(1-self.dropout)(inputs)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs