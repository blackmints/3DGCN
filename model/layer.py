from keras import initializers, regularizers, activations
from keras.engine.topology import Layer
import tensorflow as tf


class GraphEmbed(Layer):
    def __init__(self, **kwargs):
        super(GraphEmbed, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GraphEmbed, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # atoms = (samples, max_atoms, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        atoms, distances = inputs

        # Get parameters
        max_atoms = int(atoms.shape[1])
        atom_feat = int(atoms.shape[-1])
        coor_dims = int(distances.shape[-1])

        # Generate vector features filled with zeros
        vector_features = tf.zeros_like(atoms)
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, coor_dims, 1])

        return [atoms, vector_features]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], (input_shape[0][0], input_shape[0][1], input_shape[-1][-1], input_shape[0][-1])]


class GraphSToS(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphSToS, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphSToS, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[-1]
        self.w_ss = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_ss')

        self.b_ss = self.add_weight(shape=(self.filters,),
                                    name='b_ss',
                                    initializer=self.bias_initializer)

        super(GraphSToS, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        scalar_features = inputs

        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])

        # Expand scalar features to 4D
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])

        # Combine between atoms
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)

        # Linear combination
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        scalar_features = tf.matmul(scalar_features, self.w_ss) + self.b_ss
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])

        # Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1], self.filters


class GraphSToV(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphSToV, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphSToV, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        self.w_sv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_sv')

        self.b_sv = self.add_weight(shape=(self.filters,),
                                    name='b_sv',
                                    initializer=self.bias_initializer)

        super(GraphSToV, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        scalar_features, distances = inputs

        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])
        coor_dims = int(distances.shape[-1])

        # Expand scalar features to 4D
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])

        # Combine between atoms
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)

        # Apply weights
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        scalar_features = tf.matmul(scalar_features, self.w_sv) + self.b_sv
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, 1, self.filters])
        scalar_features = tf.tile(scalar_features, [1, 1, 1, coor_dims, 1])

        # Expand distances to 5D
        distances = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances = tf.tile(distances, [1, 1, 1, 1, self.filters])

        # Tensor product
        vector_features = tf.multiply(scalar_features, distances)

        # Activation
        vector_features = self.activation(vector_features)

        return vector_features

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], input_shape[1][-1], self.filters


class GraphVToV(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphVToV, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphVToV, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[-1]
        self.w_vv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vv')

        self.b_vv = self.add_weight(shape=(self.filters,),
                                    name='b_vv',
                                    initializer=self.bias_initializer)

        super(GraphVToV, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        vector_features = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # Expand vector features to 5D
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])

        # Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)

        # Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        vector_features = tf.matmul(vector_features, self.w_vv) + self.b_vv
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # Activation
        vector_features = self.activation(vector_features)

        return vector_features

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1], input_shape[-2], self.filters


class GraphVToS(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphVToS, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphVToS, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        self.w_vs = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vs')

        self.b_vs = self.add_weight(shape=(self.filters,),
                                    name='b_vs',
                                    initializer=self.bias_initializer)

        super(GraphVToS, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        vector_features, distances = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # Expand vector features to 5D
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])

        # Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)

        # Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        vector_features = tf.matmul(vector_features, self.w_vs) + self.b_vs
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # # Calculate r^ = r / |r| and expand it to 5D
        # distances_hat = tf.sqrt(tf.reduce_sum(tf.square(distances), axis=-1, keepdims=True))
        # distances_hat = distances_hat + tf.cast(tf.equal(distances_hat, 0), tf.float32)
        # distances_hat = tf.divide(distances, distances_hat)
        # distances_hat = tf.reshape(distances_hat, [-1, max_atoms, max_atoms, coor_dims, 1])
        # distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])

        distances_hat = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])

        # Projection of v onto r = v (dot) r^
        scalar_features = tf.multiply(vector_features, distances_hat)
        scalar_features = tf.reduce_sum(scalar_features, axis=-2)

        # Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1], self.filters


class GraphConvS(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.pooling = pooling

        super(GraphConvS, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphConvS, self).get_config()
        base_config['filters'] = self.filters
        base_config['pooling'] = self.pooling
        return base_config

    def build(self, input_shape):
        atom_feat_1 = input_shape[0][-1]
        atom_feat_2 = input_shape[1][-1]
        self.w_conv_scalar = self.add_weight(shape=(atom_feat_1 + atom_feat_2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_scalar')

        self.b_conv_scalar = self.add_weight(shape=(self.filters,),
                                             name='b_conv_scalar',
                                             initializer=self.bias_initializer)
        super(GraphConvS, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # scalar_features_1 = (samples, max_atoms, max_atoms, atom_feat)
        # scalar_features_2 = (samples, max_atoms, max_atoms, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        scalar_features_1, scalar_features_2, adjacency = inputs

        # Get parameters
        max_atoms = int(scalar_features_1.shape[1])
        atom_feat_1 = int(scalar_features_1.shape[-1])
        atom_feat_2 = int(scalar_features_2.shape[-1])

        # Concatenate two features
        scalar_features = tf.concat([scalar_features_1, scalar_features_2], axis=-1)

        # Linear combination
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat_1 + atom_feat_2])
        scalar_features = tf.matmul(scalar_features, self.w_conv_scalar) + self.b_conv_scalar
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])

        # Adjacency masking
        adjacency = tf.reshape(adjacency, [-1, max_atoms, max_atoms, 1])
        adjacency = tf.tile(adjacency, [1, 1, 1, self.filters])
        scalar_features = tf.multiply(scalar_features, adjacency)

        # Integrate over second atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=2)
        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=2)

        # Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.filters


class GraphConvV(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.pooling = pooling

        super(GraphConvV, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphConvV, self).get_config()
        base_config['filters'] = self.filters
        base_config['pooling'] = self.pooling
        return base_config

    def build(self, input_shape):
        atom_feat_1 = input_shape[0][-1]
        atom_feat_2 = input_shape[1][-1]
        self.w_conv_vector = self.add_weight(shape=(atom_feat_1 + atom_feat_2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_vector')

        self.b_conv_vector = self.add_weight(shape=(self.filters,),
                                             initializer=self.bias_initializer,
                                             name='b_conv_vector')
        super(GraphConvV, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # vector_features_1 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # vector_features_2 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        vector_features_1, vector_features_2, adjacency = inputs

        # Get parameters
        max_atoms = int(vector_features_1.shape[1])
        atom_feat_1 = int(vector_features_1.shape[-1])
        atom_feat_2 = int(vector_features_2.shape[-1])
        coor_dims = int(vector_features_1.shape[-2])

        # Concatenate two features
        vector_features = tf.concat([vector_features_1, vector_features_2], axis=-1)

        # Linear combination
        vector_features = tf.reshape(vector_features, [-1, atom_feat_1 + atom_feat_2])
        vector_features = tf.matmul(vector_features, self.w_conv_vector) + self.b_conv_vector
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # Adjacency masking
        adjacency = tf.reshape(adjacency, [-1, max_atoms, max_atoms, 1, 1])
        adjacency = tf.tile(adjacency, [1, 1, 1, coor_dims, self.filters])
        vector_features = tf.multiply(vector_features, adjacency)

        # Integrate over second atom axis
        if self.pooling == "sum":
            vector_features = tf.reduce_sum(vector_features, axis=2)
        elif self.pooling == "max":
            vector_features = tf.reduce_max(vector_features, axis=2)

        # Activation
        vector_features = self.activation(vector_features)

        return vector_features

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][-2], self.filters


class GraphGather(Layer):
    def __init__(self,
                 pooling="sum",
                 system="cartesian",
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.pooling = pooling
        self.system = system

        super(GraphGather, self).__init__(**kwargs)

    def build(self, inputs_shape):
        super(GraphGather, self).build(inputs_shape)

    def get_config(self):
        base_config = super(GraphGather, self).get_config()
        base_config['pooling'] = self.pooling
        base_config['system'] = self.system
        return base_config

    def call(self, inputs, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        scalar_features, vector_features = inputs

        # Get parameters
        coor_dims = int(vector_features.shape[2])
        atom_feat = int(vector_features.shape[-1])

        # Integrate over atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=1)
            vector_features = tf.reduce_sum(vector_features, axis=1)

        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=1)

            vector_features = tf.transpose(vector_features, perm=[0, 2, 3, 1])
            size = tf.sqrt(tf.reduce_sum(tf.square(vector_features), axis=1))
            idx = tf.reshape(tf.argmax(size, axis=-1, output_type=tf.int32), [-1, 1, atom_feat, 1])
            idx = tf.tile(idx, [1, coor_dims, 1, 1])
            vector_features = tf.reshape(tf.batch_gather(vector_features, idx), [-1, coor_dims, atom_feat])

        # Activation
        scalar_features = self.activation(scalar_features)
        vector_features = self.activation(vector_features)

        if self.system == "spherical":
            x, y, z = tf.unstack(vector_features, axis=1)
            r = tf.sqrt(tf.square(x) + tf.square(y) + tf.square(z))
            t = tf.acos(tf.divide(z, r + tf.cast(tf.equal(r, 0), dtype=float)))
            p = tf.atan(tf.divide(y, x + tf.cast(tf.equal(x, 0), dtype=float)))
            vector_features = tf.stack([r, t, p], axis=1)

        return [scalar_features, vector_features]

    def compute_output_shape(self, inputs_shape):
        return [(inputs_shape[0][0], inputs_shape[0][2]), (inputs_shape[1][0], inputs_shape[1][2], inputs_shape[1][3])]
