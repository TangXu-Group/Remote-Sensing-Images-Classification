import tensorflow as tf
import numpy as np
import cv2
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

vgg16_npy_path = "/home/admin324/PycharmProjects/data/vgg16.npy"
data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

weights = {
    "conv_weights1_1": tf.get_variable("vgg_w1_1", initializer=data_dict["conv1_1"][0]),
    "conv_biases1_1": tf.get_variable("vgg_b1_1", initializer=data_dict["conv1_1"][1]),

    "conv_weights1_2": tf.get_variable("vgg_w1_2", initializer=data_dict["conv1_2"][0]),
    "conv_biases1_2": tf.get_variable("vgg_b1_2", initializer=data_dict["conv1_2"][1]),

    "conv_weights2_1": tf.get_variable("vgg_w2_1", initializer=data_dict["conv2_1"][0]),
    "conv_biases2_1": tf.get_variable("vgg_b2_1", initializer=data_dict["conv2_1"][1]),

    "conv_weights2_2": tf.get_variable("vgg_w2_2", initializer=data_dict["conv2_2"][0]),
    "conv_biases2_2": tf.get_variable("vgg_b2_2", initializer=data_dict["conv2_2"][1]),

    "conv_weights3_1": tf.get_variable("vgg_w3_1", initializer=data_dict["conv3_1"][0]),
    "conv_biases3_1": tf.get_variable("vgg_b3_1", initializer=data_dict["conv3_1"][1]),

    "conv_weights3_2": tf.get_variable("vgg_w3_2", initializer=data_dict["conv3_2"][0]),
    "conv_biases3_2": tf.get_variable("vgg_b3_2", initializer=data_dict["conv3_2"][1]),

    "conv_weights3_3": tf.get_variable("vgg_w3_3", initializer=data_dict["conv3_3"][0]),
    "conv_biases3_3": tf.get_variable("vgg_b3_3", initializer=data_dict["conv3_3"][1]),

    "conv_weights4_1": tf.get_variable("vgg_w4_1", initializer=data_dict["conv4_1"][0]),
    "conv_biases4_1": tf.get_variable("vgg_b4_1", initializer=data_dict["conv4_1"][1]),

    "conv_weights4_2": tf.get_variable("vgg_w4_2", initializer=data_dict["conv4_2"][0]),
    "conv_biases4_2": tf.get_variable("vgg_b4_2", initializer=data_dict["conv4_2"][1]),

    "conv_weights4_3": tf.get_variable("vgg_w4_3", initializer=data_dict["conv4_3"][0]),
    "conv_biases4_3": tf.get_variable("vgg_b4_3", initializer=data_dict["conv4_3"][1]),

    "conv_weights5_1": tf.get_variable("vgg_w5_1", initializer=data_dict["conv5_1"][0]),
    "conv_biases5_1": tf.get_variable("vgg_b5_1", initializer=data_dict["conv5_1"][1]),

    "conv_weights5_2": tf.get_variable("vgg_w5_2", initializer=data_dict["conv5_2"][0]),
    "conv_biases5_2": tf.get_variable("vgg_b5_2", initializer=data_dict["conv5_2"][1]),

    "conv_weights5_3": tf.get_variable("vgg_w5_3", initializer=data_dict["conv5_3"][0]),
    "conv_biases5_3": tf.get_variable("vgg_b5_3", initializer=data_dict["conv5_3"][1]),


    "fc6_weights": tf.get_variable("vgg_wf1", shape=[512*2 , 4096],
                                   initializer=tf.random_normal_initializer(mean=0, stddev=0.01)),
    "fc6_biases": tf.get_variable("vgg_bf1", shape=[4096], initializer=tf.random_normal_initializer(mean=0, stddev=0)),
    "fc7_weights": tf.get_variable("vgg_wf2", shape=[4096, 4096],
                                   initializer=tf.random_normal_initializer(mean=0, stddev=0.01)),
    "fc7_biases": tf.get_variable("vgg_bf2", shape=[4096], initializer=tf.random_normal_initializer(mean=0, stddev=0)),
    "fc8_weights": tf.get_variable("vgg_wf3", shape=[4096, 21],
                                   initializer=tf.random_normal_initializer(mean=0, stddev=0.01)),
    "fc8_biases": tf.get_variable("vgg_bf3", shape=[21], initializer=tf.random_normal_initializer(mean=0, stddev=0)),

}


def VGG16(x, weights,train_phase):
    # 1 convolutional layer
    conv1_1 = conv_layer(x, "conv_weights1_1", "conv_biases1_1", "conv1_1", weights)
    conv1_2 = conv_layer(conv1_1, "conv_weights1_2", "conv_biases1_2", "conv1_2", weights)
    pool1 = max_pool(conv1_2, "pool1")

    # 2 convolutional layer
    conv2_1 = conv_layer(pool1, "conv_weights2_1", "conv_biases2_1", "conv2_1", weights)
    conv2_2 = conv_layer(conv2_1, "conv_weights2_2", "conv_biases2_2", "conv2_2", weights)
    pool2 = max_pool(conv2_2, "pool2")
    # 3 convolutional layer
    conv3_1 = conv_layer(pool2, "conv_weights3_1", "conv_biases3_1", "conv3_1", weights)
    conv3_2 = conv_layer(conv3_1, "conv_weights3_2", "conv_biases3_2", "conv3_2", weights)
    conv3_3 = conv_layer(conv3_2, "conv_weights3_3", "conv_biases3_3", "conv3_3", weights)
    pool3 = max_pool(conv3_3, "pool3")
    # 4 convolutional layer
    conv4_1 = conv_layer(pool3, "conv_weights4_1", "conv_biases4_1", "conv4_1", weights)
    conv4_2 = conv_layer(conv4_1, "conv_weights4_2", "conv_biases4_2", "conv4_2", weights)
    conv4_3 = conv_layer(conv4_2, "conv_weights4_3", "conv_biases4_3", "conv4_3", weights)
    pool4 = max_pool(conv4_3, "pool4")
    # 5 convolutional layer
    conv5_1 = conv_layer(pool4, "conv_weights5_1", "conv_biases5_1", "conv5_1", weights)
    conv5_2 = conv_layer(conv5_1, "conv_weights5_2", "conv_biases5_2", "conv5_2", weights)
    conv5_3 = conv_layer(conv5_2, "conv_weights5_3", "conv_biases5_3", "conv5_3", weights)
    pool5 = max_pool(conv5_3, "pool5")

    SP_pool5,mask_SP = spatial_attention(pool5)
    SP_pool5 = bn(SP_pool5, train_phase, "BN_SP")
    mask_SP = tf.nn.relu(mask_SP)

    SE_pool5 = SEblock(pool5, 16, "SE")
    SE_pool5 = bn(SE_pool5, train_phase, "BN_SE")

    mask_SE = tf.reduce_mean(SE_pool5, axis=3)
    mask_SE = tf.nn.relu(mask_SE)
    mask_SE = tf.expand_dims(mask_SE, -1)

    return SE_pool5,SP_pool5,mask_SE,mask_SP



def fc(SE_pool5,SP_pool5,weights,kp):

    flatten = tf.concat([tf.reduce_mean(SE_pool5,[1,2]),tf.reduce_mean(SP_pool5,[1,2])],axis=1)


    dense1_gap = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten, weights["fc6_weights"]), weights["fc6_biases"]))
    dense1_gap = tf.nn.dropout(dense1_gap, keep_prob=kp)

    dense2_gap = tf.nn.relu(tf.nn.bias_add(tf.matmul(dense1_gap, weights["fc7_weights"]), weights["fc7_biases"]))
    dense2_gap = tf.nn.dropout(dense2_gap, keep_prob=kp)

    label_logits_gap = tf.nn.bias_add(tf.matmul(dense2_gap, weights["fc8_weights"]), weights["fc8_biases"])

    return label_logits_gap



def conv_layer(bottom, weight_name, biase_name, name, data_dict):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(bottom, data_dict[weight_name], [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, data_dict[biase_name])
        relu = tf.nn.relu(bias)
        return relu


def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def read_txt(train_path, test_path):
    train_filename = []
    train_filelabel = []
    with open(train_path, 'r') as f:
        x = f.readlines()
        for name in x:
            train_filename.append(name.strip().split()[0])
            train_filelabel.append(int(name.strip().split()[1]))
    train_filename = np.array(train_filename)
    train_filelabel = np.array(train_filelabel)

    test_filename = []
    test_filelabel = []
    with open(test_path, 'r') as f:
        x = f.readlines()
        for name in x:
            test_filename.append(name.strip().split()[0])
            test_filelabel.append(int(name.strip().split()[1]))
    test_filename = np.array(test_filename)
    test_filelabel = np.array(test_filelabel)

    return train_filename, train_filelabel, test_filename, test_filelabel



def read_data(path):
    data = []

    for i in range(path.shape[0]):
        temp = cv2.imread(path[i])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = cv2.resize(temp, dsize=(224, 224))
        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp)) - 0.5

        data.append(temp)
    data = np.array(data)

    return data  #



def read_data_180(path):

    data = []

    for i in range(path.shape[0]):
        temp = cv2.imread(path[i])
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        temp = cv2.resize(temp, dsize=(224, 224))
        temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp)) - 0.5

        temp = np.rot90(temp, 2)
        data.append(temp)

    data = np.array(data)

    return data  # read txt and read data



def SEblock(input_x, ratio, name):
    out_dim = input_x.shape.as_list()[-1]
    with tf.name_scope(name) as scope:
        squeeze = tf.reduce_mean(input_x, [1, 2])
        weight1 = tf.Variable(tf.truncated_normal([out_dim, int(out_dim / ratio)],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name=name + 'weight1')
        excitation = tf.matmul(squeeze, weight1)
        excitation = tf.nn.relu(excitation)
        weight2 = tf.Variable(tf.truncated_normal([int(out_dim / ratio), out_dim],
                                                  dtype=tf.float32,
                                                  stddev=1e-1), name=name + 'weight2')
        excitation = tf.matmul(excitation, weight2)
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = input_x * excitation

        return scale




def spatial_attention(feature_map, K=1024, scope="SP", reuse=None):
    """This method is used to add spatial attention to model.

    Parameters
    ---------------
    @feature_map: Which visual feature map as branch to use.
    @K: Map `H*W` units to K units. Now unused.
    @reuse: reuse variables if use multi gpus.

    Return
    ---------------
    @attended_fm: Feature map with Spatial Attention.
    """
    with tf.variable_scope(scope, 'SpatialAttention', reuse=reuse):
        # Tensorflow's tensor is in BHWC format. H for row split while W for column split.
        H, W, C = 7,7,512
        w_s = tf.get_variable("SpatialAttention_w_s", [C, 1],
                              dtype=tf.float32,
                              initializer=tf.initializers.orthogonal,
                              )
        b_s = tf.get_variable("SpatialAttention_b_s", [1],
                              dtype=tf.float32,
                              initializer=tf.initializers.zeros)
        spatial_attention_fm = tf.matmul(tf.reshape(feature_map, [-1, C]), w_s) + b_s
        spatial_attention_fm = tf.nn.sigmoid(tf.reshape(spatial_attention_fm, [-1, W * H]))

        attention = tf.reshape(tf.concat([spatial_attention_fm] * C, axis=1), [-1, H, W, C])
        attended_fm = attention * feature_map


        spatial_attention_fm = tf.reshape(spatial_attention_fm, [-1, 7, 7])
        spatial_attention_fm = tf.expand_dims(spatial_attention_fm, -1)

        return attended_fm, spatial_attention_fm



def bn(x,train_phase,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None, # is this right?
    trainable=True,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True, # is this right?
    trainable=True,
    scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)

    return z
