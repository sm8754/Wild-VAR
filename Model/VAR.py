from ..Checkpoint import parameters
import numpy as np
import tensorflow as tf

# Define the rotation tensor function
def rotation_tensor(theta, phi, psi, b=1):
    one = tf.ones((b, 1, 1))
    zero = tf.zeros((b, 1, 1))

    rot_x = tf.concat([
        tf.concat([one, zero, zero], axis=1),
        tf.concat([zero, tf.cos(theta), tf.sin(theta)], axis=1),
        tf.concat([zero, -tf.sin(theta), tf.cos(theta)], axis=1),
    ], axis=2)

    rot_y = tf.concat([
        tf.concat([tf.cos(phi), zero, -tf.sin(phi)], axis=1),
        tf.concat([zero, one, zero], axis=1),
        tf.concat([tf.sin(phi), zero, tf.cos(phi)], axis=1),
    ], axis=2)

    rot_z = tf.concat([
        tf.concat([tf.cos(psi), -tf.sin(psi), zero], axis=1),
        tf.concat([tf.sin(psi), tf.cos(psi), zero], axis=1),
        tf.concat([zero, zero, one], axis=1)
    ], axis=2)

    return tf.matmul(rot_z, tf.matmul(rot_y, rot_x))

def cubing(clip_X,output_channel, mode, num_cameras):

    with tf.variable_scope('Cubing', reuse=tf.AUTO_REUSE):
        # First convolutional layer
        conv1 = tf.layers.conv2d(clip_X, 32, (3, 3), strides=(1, 1), padding='same',
                                 activation=None, use_bias=False, name='conv1')
        conv_bn1 = tf.layers.batch_normalization(conv1, training=mode, name='conv_bn_3x3_1')
        activation1 = tf.nn.relu(conv_bn1, name='relu1')

        # Second convolutional layer
        conv2 = tf.layers.conv2d(activation1, 96 + 3, (3, 3), strides=(1, 1), padding='same',
                                 activation=None, use_bias=False, name='conv2')
        conv_bn2 = tf.layers.batch_normalization(conv2, training=mode, name='conv_bn_3x3_2')
        activation2 = tf.nn.relu(conv_bn2, name='relu2')

        # Apply the CameraProps logic
        x = activation2
        cam = tf.layers.conv2d(x, 128, 3)
        cam_relu = tf.nn.relu(cam)
        cam_mean = tf.reduce_mean(cam_relu, axis=[1, 2])
        cam2 = tf.layers.dense(cam_mean, 32)
        cam2_relu = tf.nn.relu(cam2)
        b = tf.shape(cam2_relu)[0]
        r = tf.layers.dense(cam2_relu, 3)
        rot = rotation_tensor(r[:, 0], r[:, 1], r[:, 2], b)
        trans = tf.reshape(tf.layers.dense(cam2_relu, 3), (b, 3, 1, 1))

        # Define CameraProjection within the scope
        cameras = [tf.Variable(tf.random.uniform((4,), -1, 1)) for _ in range(num_cameras)]
        cam_rot = [tf.Variable(tf.random.uniform((3,), 0, np.pi)) for _ in range(num_cameras)]

        fts = x[:, :-3]
        pt = x[:, -3:]
        pw = tf.einsum('bphw,bpq->bqhw', pt, rot)
        pw += trans

        views = []
        projs = []
        for r, c in zip(cam_rot, cameras):
            rot_cam = rotation_tensor(r[0], r[1], r[2], 1)
            cam_pt = tf.einsum('bphw,pq->bghw', pw, tf.squeeze(rot_cam, 0))
            proj = tf.stack([(cam_pt[:, 0] * c[0] + c[2]), (cam_pt[:, 1] * c[1] + c[3])], axis=-1)
            proj = tf.tanh(proj)
            views.append(tf.image.resize(fts, proj))
            projs.append(proj)

        return tf.concat(views, axis=1),tf.concat(projs, axis=1)

def encoder(clip_X, mode):
    input_channel = 32
    # t, c,  n, s
    arguments = [[1, 16, 1, 1],
                 [6, 32, 3, 2],
                 [6, 64, 4, 2],
                 [6, 96, 3, 2],
                 [6, 160, 1, 1]]

    # building first layer
    with tf.variable_scope('Conv2d3x3', reuse=None):
        conv_3x3 = tf.layers.conv2d(clip_X,
                                    input_channel, (3, 3),
                                    strides=(2, 2),
                                    padding='same',
                                    activation=None,
                                    use_bias=False,
                                    name='conv')
        conv_bn_3x3 = tf.layers.batch_normalization(conv_3x3, training=mode, name='conv_bn_3x3')
        activation = tf.nn.relu(conv_bn_3x3, name='relu')
    # building inverted residual blocks
    m = 0
    with tf.variable_scope('Bottlenecks', reuse=None):
        for t, c, n, s in arguments:
            m += 1

            with tf.variable_scope('Blocks' + str(m), reuse=None):
                output_channel = int(c)

                for i in range(n):
                    with tf.variable_scope('block' + str(i), reuse=None):
                        filters = t * input_channel

                        if i == 0:
                            # pointwise layer
                            pw = tf.layers.conv2d(activation,
                                                  filters, (1, 1),
                                                  strides=(1, 1),
                                                  padding='valid',
                                                  activation=None,
                                                  use_bias=False,
                                                  name='conv_non-linear')
                            pw_bn = tf.layers.batch_normalization(pw, training=mode, name='pw_bn')
                            pw_relu = tf.nn.relu(pw_bn, name='pw_relu')

                            # depthwise layer
                            dw = tf.contrib.layers.separable_conv2d(pw_relu, num_outputs=None,
                                                                    kernel_size=[3, 3],
                                                                    depth_multiplier=1,
                                                                    stride=[s, s],
                                                                    padding='SAME',
                                                                    activation_fn=None,
                                                                    biases_initializer=None)
                            dw_bn = tf.layers.batch_normalization(dw, training=mode, name='dw_bn')
                            dw_relu = tf.nn.relu(dw_bn, name='dw_relu')

                            # pointwise linear layer
                            plw = tf.layers.conv2d(dw_relu,
                                                   output_channel, (1, 1),
                                                   strides=(1, 1),
                                                   padding='valid',
                                                   activation=None,
                                                   use_bias=False,
                                                   name='conv_linear')
                            pwl_bn = tf.layers.batch_normalization(plw, training=mode, name='pwl_bn')

                            # residual connection
                            if (s == 1) and (filters == output_channel):
                                activation = pwl_bn + activation
                            else:
                                activation = pwl_bn
                        else:
                            # pointwise layer
                            pw = tf.layers.conv2d(activation,
                                                  filters, (1, 1),
                                                  strides=(1, 1),
                                                  padding='valid',
                                                  activation=None,
                                                  use_bias=False,
                                                  name='conv_non-linear')
                            pw_bn = tf.layers.batch_normalization(pw, training=mode, name='pw_bn')
                            pw_relu = tf.nn.relu(pw_bn, name='pw_relu')

                            # depthwise layer
                            dw = tf.contrib.layers.separable_conv2d(pw_relu, num_outputs=None,
                                                                    kernel_size=[3, 3],
                                                                    depth_multiplier=1,
                                                                    stride=[1, 1],
                                                                    padding='SAME',
                                                                    activation_fn=None,
                                                                    biases_initializer=None)
                            dw_bn = tf.layers.batch_normalization(dw, training=mode, name='dw_bn')
                            dw_relu = tf.nn.relu(dw_bn, name='dw_relu')

                            # pointwise linear layer
                            plw = tf.layers.conv2d(dw_relu,
                                                   output_channel, (1, 1),
                                                   strides=(1, 1),
                                                   padding='valid',
                                                   activation=None,
                                                   use_bias=False,
                                                   name='conv_linear')
                            pwl_bn = tf.layers.batch_normalization(plw, training=mode, name='pwl_bn')

                            # residual connection
                            if (s == 1) and (filters == output_channel):
                                activation = pwl_bn + activation
                            else:
                                activation = pwl_bn
                        input_channel = output_channel

    return activation


class PatchToEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, embedding_dim):
        super(PatchToEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.projection = tf.keras.layers.Dense(embedding_dim)

    def call(self, feature_maps):
        batch_size = tf.shape(feature_maps)[0]

        patches = tf.image.extract_patches(
            images=feature_maps,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [batch_size, -1, tf.shape(patches)[-1]])

        embeddings = self.projection(patches)
        return embeddings


class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches, embedding_dim),
            initializer='random_normal',
            trainable=True
        )

    def call(self, embeddings):
        return embeddings + self.position_embeddings

def self_attention(Tensor_q, Tensor_k, Tensor_v, d_k):
    Tensor_k_T = tf.transpose(Tensor_k, perm=[1, 0], name='transpose')
    scale = tf.div(tf.matmul(Tensor_q, Tensor_k_T), tf.sqrt(tf.cast(d_k, dtype=tf.float32)), name='scale')
    attention = tf.nn.softmax(scale, axis=-1, name='attention')
    head_output = tf.matmul(attention, Tensor_v, name='head_output')
    return head_output


def multi_head_attention(output, d_k, d_v, d_model, num_head):
    outputs = []
    attens = []
    weight_init = tf.truncated_normal_initializer(stddev=0.01)
    for i in range(num_head):
        # Projections matrices
        w_q = tf.get_variable(name='w_q/' + str(i), shape=[d_model, d_k], initializer=weight_init)
        w_k = tf.get_variable(name='w_k/' + str(i), shape=[d_model, d_k], initializer=weight_init)
        w_v = tf.get_variable(name='w_v/' + str(i), shape=[d_model, d_v], initializer=weight_init)

        # Linear
        Tensor_q_linear = tf.matmul(output, w_q, name='Tensor_q_linear/' + str(i))
        Tensor_k_linear = tf.matmul(output, w_k, name='Tensor_k_linear/' + str(i))
        Tensor_v_linear = tf.matmul(output, w_v, name='Tensor_v_linear/' + str(i))

        # Self attention
        head_output = self_attention(Tensor_q_linear, Tensor_k_linear, Tensor_v_linear, d_k)
        outputs.append(head_output)

        #Mulihead-attention
        attention_scores = (tf.nn.tanh(Tensor_q_linear) + 1)* (tf.transpose(tf.nn.tanh(Tensor_k_linear)) + 1)
        atten =  attention_scores @ Tensor_v_linear
        attens.append(atten)

    concat = tf.concat(outputs, axis=1, name='concat')

    # Linear
    w_o = tf.get_variable(name='w_o', shape=[num_head * d_v, d_model], initializer=weight_init)
    multi_head_output = tf.matmul(concat, w_o, name='multi_head_output')

    concat = tf.concat(attens, axis=1, name='concat')

    # Linear
    w_a = tf.get_variable(name='w_a', shape=[num_head * d_v, d_model], initializer=weight_init)
    multi_head_atten = tf.matmul(concat, w_a, name='multi_head_output')

    multi_head_output = tf.add(multi_head_output,multi_head_atten)

    multi_head_output = tf.expand_dims(multi_head_output, 0, name='expand_dims')

    return multi_head_output


def ffn(multi_head_output, d_ff, d_model):
    ffn_conv1 = tf.layers.conv1d(multi_head_output, d_ff, 1, activation=tf.nn.relu, use_bias=True, name='ffn_conv1')
    ffn_conv2 = tf.layers.conv1d(ffn_conv1, d_model, 1, activation=None, use_bias=True, name='ffn_conv2')
    residual_output = tf.add(multi_head_output, ffn_conv2, name='residual_output')
    residual_output = tf.squeeze(residual_output, name='squeeze')
    return residual_output


def decoder(output, d_k, d_v, d_model, num_head, d_ff,num_blocks):
    with tf.variable_scope('Decodeblocks', reuse=None):
        for i in range(num_blocks):
            with tf.variable_scope('Blocks' + str(i), reuse=None):
                multi_head_output = multi_head_attention(output, d_k, d_v, d_model, num_head)
                output = ffn(multi_head_output, d_ff, d_model)

    return output

def var(clip_X, mode):
    # Encoder
    with tf.variable_scope('encoder', reuse=None):
        output = encoder(clip_X, mode)

    with tf.variable_scope('Conv2d1x1', reuse=None):
        conv_1x1 = tf.layers.conv2d(output,
                                    parameters.d_model, (1, 1),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation=None,
                                    use_bias=False,
                                    name='conv')
        conv_bn_1x1 = tf.layers.batch_normalization(conv_1x1, training=mode, name='conv_bn_1x1')
        activation = tf.nn.relu(conv_bn_1x1, name='relu')

        shape = activation.get_shape().as_list()
        width = shape[1]
        height = shape[2]
        output = tf.layers.average_pooling2d(activation,
                                             pool_size=(height, width),
                                             strides=(1, 1),
                                             padding='valid',
                                             name='glo_avg_epool2d')
        output = tf.squeeze(output, name='squeeze')

        # building Cubing block
    with tf.variable_scope('cubing', reuse=None):
        V_pairs,c_matrices = cubing(clip_X, parameters.d_model/2, mode, num_cameras=parameters.num_cameras)

    # Decoder
    with tf.variable_scope('decoder', reuse=None):
        for i in range(parameters.num_stacks):
            with tf.variable_scope('decoder' + str(i), reuse=None):
                V_pairs = PatchToEmbedding(patch_size=1, embedding_dim=parameters.d_model)(V_pairs)
                output = tf.concat([output, V_pairs], axis=0)
                num_patches = (output.shape[1]) * (output.shape[2])
                output = PositionEmbedding(num_patches=num_patches, embedding_dim=parameters.d_model)(output)
                output = decoder(output,
                                 parameters.d_k,
                                 parameters.d_v,
                                 parameters.d_model,
                                 parameters.num_head,
                                 parameters.d_ff,
                                 parameters.num_blocks)

        clip_logits = tf.reduce_mean(output, axis=1, name='clip_logits')
        clip_logits = tf.expand_dims(clip_logits, 0, name='expand_dims')

    # Classifier
    with tf.variable_scope('classifier', reuse=None):
        logits = tf.layers.dense(clip_logits, parameters.NUM_CLASSESS, activation=None, use_bias=True, name='logits')
        softmax_output = tf.nn.softmax(logits, name='softmax_output')

    return logits, softmax_output,V_pairs,c_matrices,clip_logits

