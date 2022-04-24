import tensorflow as tf
import os
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import GRU, TimeDistributed, Dense, Reshape


def mnloss(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


from collections import namedtuple

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'padding', 'activation', 'bn'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth', 'padding', 'activation', 'bn', 'dia_rate'])
MaxPooling = namedtuple('MaxPooling', ['poolsize'])
GRUlayer = namedtuple('GRUlayer', ['return_sequences', 'go_backwards', 'kernel_initializer'])
BottleNeck = namedtuple('BottleNeck', ['kernel', 'stride', 'depth', 'padding', 't_rate', 'n_rate'])
GRPConv = namedtuple('GRPConv', ['kernel', 'stride', 'depth', 'padding', 'activation', 'group', 'bn'])
ResConv = namedtuple('ResConv', ['kernel', 'stride', 'depth', 'padding', 'activation', 'group', 'bn'])
actv_str = 'selu'
kernel_ini = 'lecun_normal'  # 'lecun_normal'


def create_mobilenet_model(input_window_length, mb=False):
    alpha = 1
    beta = 1
    # Network Parameters
    nb_filters = [128, 64, 64, 64, 64, 64, 64, 64, 128]
    nb_filters = [i * 2 for i in nb_filters]
    depth = 9
    stacks = 1
    residual = False
    use_bias = True
    dropout = 0
    mask = True
    res_l2 = 0.0
    bn = False
    grp = 4
    MOBILENETV1_CONV_DEFS = [
        Conv(kernel=8, stride=1, depth=30, padding='same', activation=actv_str, bn=False),
        DepthSepConv(kernel=10, stride=1, depth=30, padding='same', activation=actv_str, bn=False, dia_rate=1),
        DepthSepConv(kernel=8, stride=1, depth=30, padding='same', activation=actv_str, bn=False, dia_rate=1),
        DepthSepConv(kernel=6, stride=1, depth=40, padding='same', activation=actv_str, bn=False, dia_rate=1),
        DepthSepConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, bn=False, dia_rate=1),
        DepthSepConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, bn=False, dia_rate=1),
        # DepthSepConv(kernel=3,  stride=1, depth=64  ,padding='same', activation=actv_str, bn=False),
    ]
    GRP_CONV_DEFS = [
        Conv(kernel=9, stride=1, depth=32, padding='same', activation=actv_str, bn=True),
        GRPConv(kernel=9, stride=1, depth=32, padding='same', activation=actv_str, group=grp, bn=bn),
        GRPConv(kernel=7, stride=1, depth=32, padding='same', activation=actv_str, group=grp, bn=bn),
        GRPConv(kernel=7, stride=1, depth=48, padding='same', activation=actv_str, group=grp, bn=bn),
        GRPConv(kernel=5, stride=1, depth=48, padding='same', activation=actv_str, group=grp, bn=bn),
        GRPConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, group=grp, bn=bn),
        GRPConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, group=grp, bn=bn),
        GRPConv(kernel=5, stride=1, depth=96, padding='same', activation=actv_str, group=grp, bn=bn),
        Conv(kernel=1, stride=1, depth=64, padding='same', activation=actv_str, bn=False),
        # DepthSepConv(kernel=input_window_length, stride=1, depth=1  ,padding='valid', activation=actv_str, bn=False, dia_rate=1),
    ]
    Res_CONV_DEFS = [
        Conv(kernel=9, stride=1, depth=32, padding='same', activation=actv_str, bn=True),
        ResConv(kernel=5, stride=1, depth=32, padding='same', activation=actv_str, group=grp, bn=bn),
        ResConv(kernel=5, stride=1, depth=48, padding='same', activation=actv_str, group=grp, bn=bn),
        ResConv(kernel=5, stride=1, depth=48, padding='same', activation=actv_str, group=grp, bn=bn),
        ResConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, group=grp, bn=bn),
        ResConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, group=grp, bn=bn),
        ResConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, group=grp, bn=bn),
        ResConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, group=grp, bn=bn),
        ResConv(kernel=5, stride=1, depth=64, padding='same', activation=actv_str, group=grp, bn=bn),
    ]
    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
    if mb:
        net = create_seperate_layer(MOBILENETV1_CONV_DEFS, reshape_layer, alpha=alpha)
    else:
        net = create_seperate_layer(Res_CONV_DEFS, reshape_layer, alpha=alpha)
    # ----------------------depth seperate dense layer-----------------------------#
    # net = tf.keras.layers.GlobalAveragePooling2D()(net)

    net = tf.keras.layers.SeparableConv2D(filters=int(512 * beta), kernel_size=(1, input_window_length),
                                          kernel_initializer=kernel_ini, strides=1, padding='valid',
                                          activation=actv_str)(net)  #
    # net = tf.keras.layers.Dropout(0.25)(net)
    output_layer = tf.keras.layers.Flatten()(net)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(output_layer)
    # net = tf.keras.layers.Dropout(0.25)(net)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def create_dense_model(input_window_length, mb=True):
    bn = False
    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    net0 = net = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
    net = tf.keras.layers.Conv2D(32, 5, activation=actv_str, padding='same', kernel_initializer=kernel_ini)(net)
    dense1 = net = dense_block(net, grow_rate=24, bn=bn, n=8, kernel=7)
    net2 = net = tf.keras.layers.Conv2D(64, 1, activation=actv_str, kernel_initializer=kernel_ini)(net)
    dense2 = net = dense_block(net, grow_rate=32, bn=bn, n=6, kernel=5)
    net = tf.keras.layers.Conv2D(64, 1, activation=actv_str, kernel_initializer=kernel_ini)(net)
    # ----------------------depth seperate dense layer-----------------------------#
    net = tf.keras.layers.SeparableConv2D(filters=512, kernel_size=(1, input_window_length),
                                          kernel_initializer=kernel_ini, strides=1, padding='valid',
                                          activation=actv_str)(net)  #
    # net = tf.keras.layers.GlobalAveragePooling2D()(net)
    output_layer = tf.keras.layers.Flatten()(net)
    output_layer = tf.keras.layers.Dense(1, activation="linear", name='output')(output_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def SeBlock(inputs):
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1]) // 4, use_bias=False, activation=tf.keras.activations.relu)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]), use_bias=False, activation=tf.keras.activations.hard_sigmoid)(x)
    return tf.keras.layers.Multiply()([inputs, x])  # 给通道加权重


def create_seperate_layer(conv_defs, layer_input, alpha):
    net = layer_input
    for i, conv_def in enumerate(conv_defs):
        if isinstance(conv_def, Conv):
            net = tf.keras.layers.Convolution2D(filters=int(conv_def.depth * alpha), kernel_initializer=kernel_ini,
                                                kernel_size=(conv_def.kernel, 1),  #
                                                strides=conv_def.stride, padding=conv_def.padding,
                                                activation=conv_def.activation)(net)
            if conv_def.bn == True:
                net = tf.keras.layers.BatchNormalization()(net)

        elif isinstance(conv_def, DepthSepConv):
            net = tf.keras.layers.SeparableConv2D(filters=int(conv_def.depth * alpha), kernel_initializer=kernel_ini,
                                                  kernel_size=(1, conv_def.kernel),  #
                                                  strides=conv_def.stride, padding=conv_def.padding,
                                                  activation=conv_def.activation, dilation_rate=conv_def.dia_rate)(net)
            '''net = DepthSeparableConv(net,filters=int(conv_def.depth*alpha), kernel_initializer=kernel_ini, kernel_size=(conv_def.kernel,1),
                        strides=conv_def.stride, padding=conv_def.padding, activation=conv_def.activation, dilation_rate=conv_def.dia_rate)'''
            if conv_def.bn == True:
                net = tf.keras.layers.BatchNormalization()(net)
        elif isinstance(conv_def, MaxPooling):
            net = tf.keras.layers.MaxPooling2D(pool_size=(1, conv_def.poolsize))(net)
        elif isinstance(conv_def, BottleNeck):
            # net = _inverted_residual_block(net, conv_def.depth, (conv_def.kernel, 1), t=conv_def.t_rate, strides=conv_def.stride, n=conv_def.n_rate)
            net = bottleneck(inputs=net, filters=conv_def.depth, kernel=(1, conv_def.kernel), t=conv_def.t_rate,
                             strides=conv_def.stride, n=conv_def.n_rate)
        elif isinstance(conv_def, GRPConv):
            x = tf.keras.layers.Convolution2D(filters=conv_def.depth, kernel_size=(1, 1),
                                              activation=conv_def.activation)(net)
            # x = Conv2D_BN_ReLU(net, channel=conv_def.depth, kernel_size=(1, 1))
            x = group_conv(x=x, filters=conv_def.depth // 2, kernel=(1, conv_def.kernel), stride=conv_def.stride,
                           groups=conv_def.group, activation=conv_def.activation, bn=conv_def.bn)
            # if conv_def.bn==True:
            # x = tf.keras.layers.BatchNormalization()(x)
            x = SeBlock(x)
            net = tf.concat([net, x], axis=3)
            net = channel_shuffle(net, conv_def.group)

        elif isinstance(conv_def, ResConv):  #
            dep = int(conv_def.depth * alpha)
            shortcut = net
            x = net
            x = tf.keras.layers.SeparableConv2D(filters=dep, kernel_size=(1, conv_def.kernel), strides=conv_def.stride,
                                                padding=conv_def.padding, activation=conv_def.activation)(x)
            if conv_def.bn == True:
                x = tf.keras.layers.BatchNormalization()(x)
            # x = SeBlock(x)
            if shortcut.shape[-1] != conv_def.depth:
                shortcut = tf.keras.layers.Conv2D(dep, 1, padding='same', activation='relu')(shortcut)
            net = tf.add(shortcut, x)
    return net


def create_GRU_model(input_window_length):
    k = 64
    rsbool = True
    conv_fltr = 64
    min_seq = 8
    MAX_LEN = input_window_length + 1
    NUM_FILTERS = 64
    merge_mode = None
    input1 = tf.keras.layers.Input(shape=(MAX_LEN // k, conv_fltr))
    embed = input1
    gru1 = tf.keras.layers.GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation='tanh', return_sequences=False)(
        embed)
    Encoder1 = tf.keras.Model(input1, gru1)

    input2 = tf.keras.layers.Input(shape=(min_seq, MAX_LEN // k, conv_fltr))
    embed2 = tf.keras.layers.TimeDistributed(Encoder1)(input2)
    embed2 = tf.keras.layers.Reshape((8, conv_fltr))(embed2)
    gru2 = tf.keras.layers.GRU(NUM_FILTERS, recurrent_activation='sigmoid', activation='tanh', return_sequences=False)(
        embed2)
    Encoder2 = tf.keras.Model(input2, gru2)

    input3 = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((input_window_length, 1))(input3)
    x = tf.keras.layers.Conv1D(NUM_FILTERS, 5, activation='selu', input_shape=(input_window_length, 1), padding="same",
                               strides=1)(reshape_layer)
    x = tf.keras.layers.Reshape((min_seq, 8, MAX_LEN // k, conv_fltr))(x)
    embed3 = tf.keras.layers.TimeDistributed(Encoder2)(x)
    embed3 = tf.keras.layers.Reshape((8, conv_fltr))(embed3)  # min_seq *
    gru3 = tf.keras.layers.GRU(NUM_FILTERS * 2, recurrent_activation='sigmoid', activation='tanh',
                               return_sequences=False)(embed3)
    preds = tf.keras.layers.Dense(128, activation='selu')(gru3)

    output_layer = tf.keras.layers.Flatten()(preds)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(output_layer)
    model = tf.keras.Model(input3, output_layer)

    return model


def DepthSeparableConv(net, filters, kernel_size, strides=1, padding='same', activation='relu', dilation_rate=1,
                       kernel_initializer='he_normal'):
    net = tf.keras.layers.DepthwiseConv2D(kernel_initializer=kernel_initializer, kernel_size=kernel_size,
                                          strides=strides, padding=padding, dilation_rate=dilation_rate,
                                          use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(actv_str)(net)
    net = tf.keras.layers.Convolution2D(filters=filters, kernel_size=(1, 1), strides=1, padding='same')(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation(actv_str)(net)
    return net


def dense_block(net, n=6, kernel=5, stride=1, activation='relu', bn=True, grow_rate=32, init='he_normal'):
    x = net
    for i in range(n):
        x = dense_conv(x, init, kernel, stride, activation, bn, grow_rate)

    return x


def dense_conv(lin, init, kernel, stride, activation='relu', bn=True, grow_rate=32):
    x = lin
    # x = tf.keras.layers.Conv2D(grow_rate*2, 1, activation=activation)(x)
    # x = DepthSeparableConv(x, filters=grow_rate,kernel_size=(1,kernel))

    x = tf.keras.layers.SeparableConv2D(grow_rate, (1, kernel), strides=1, padding='same')(x)
    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    # x = SeBlock(x)
    x = tf.concat([lin, x], axis=-1)
    return x


def group_conv(x, filters, kernel, stride, groups, activation='relu', bn=True):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]

    # number of input channels per group
    nb_ig = in_channels // groups
    # number of output channels per group
    nb_og = filters // groups

    gc_list = []
    # Determine whether the number of filters is divisible by the number of groups
    assert filters % groups == 0

    for i in range(groups):
        if channel_axis == -1:
            x_group = tf.keras.layers.Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = tf.keras.layers.Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        gc_list.append(
            tf.keras.layers.Conv2D(filters=nb_og, kernel_size=kernel, strides=stride, padding='same', use_bias=True,
                                   activation=activation, kernel_initializer='he_normal')(x_group))

    return tf.keras.layers.Concatenate(axis=channel_axis)(gc_list)


def channel_shuffle(inputs, group):
    """Shuffle the channel
    Args:
        inputs: 4D Tensor
        group: int, number of groups
    Returns:
        Shuffled 4D Tensor
    """
    in_shape = inputs.get_shape().as_list()
    h, w, in_channel = in_shape[1:]
    assert in_channel % group == 0
    l = tf.reshape(inputs, [-1, h, w, in_channel // group, group])
    l = tf.transpose(l, [0, 1, 2, 4, 3])
    l = tf.reshape(l, [-1, h, w, in_channel])

    return l


def ShuffleNetv2(input_window_length):
    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    training = True
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
    x = Conv2D_BN_ReLU(inputs=reshape_layer, channel=32, kernel_size=(1, 8), stride=1, training=training)
    x = ShufflenetUnit2(inputs=x, in_channel=32, out_channel=48, training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=48, kernel_size=(1, 8), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=48, kernel_size=(1, 8), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=48, kernel_size=(1, 6), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=48, kernel_size=(1, 6), training=training)
    x = ShufflenetUnit2(inputs=x, in_channel=48, out_channel=64, training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=64, kernel_size=(1, 6), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=64, kernel_size=(1, 6), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=64, kernel_size=(1, 5), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=64, kernel_size=(1, 5), training=training)
    x = ShufflenetUnit2(inputs=x, in_channel=64, out_channel=96, training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=96, kernel_size=(1, 3), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=96, kernel_size=(1, 3), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=96, kernel_size=(1, 3), training=training)
    x = ShufflenetUnit1(inputs=x, out_channel=96, kernel_size=(1, 3), training=training)
    x = tf.keras.layers.SeparableConv2D(filters=1024, kernel_initializer='he_normal',
                                        kernel_size=(1, input_window_length), strides=1, padding='valid',
                                        activation='relu')(x)
    # x = tf.keras.layers.DepthwiseConv2D(kernel_size=(1, input_window_length), strides=1,depth_multiplier=1,padding="valid", use_bias=False, activation='relu')(x)
    # x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1,1), strides=1, activation='relu')(x)
    # x = group_conv(x,1024,(1,1),1,2)
    label_layer = tf.keras.layers.Flatten()(x)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def Shufflenet1(inputs, out_channel, kernel_size=(1, 5), training=False):
    """The unit of shufflenetv2 for stride=1
    Args:
        out_channel: int, number of channels
    """
    assert out_channel % 2 == 0
    shortcut, x = tf.split(inputs, 2, axis=3)
    x = Conv2D_BN_ReLU(x, channel=out_channel // 2, kernel_size=(1, 1), stride=1)
    x = group_conv(x=x, filters=out_channel // 2, kernel=kernel_size, stride=1, groups=4)

    x = tf.concat([shortcut, x], axis=3)
    x = channel_shuffle(x, 4)
    return x


def Shufflenet2(inputs, in_channel, out_channel, kernel_size=(1, 5), stride=1, training=False):
    """The unit of shufflenetv2 for stride=2"""

    assert out_channel % 2 == 0
    shortcut, x = inputs, inputs
    x = Conv2D_BN_ReLU(x, channel=out_channel // 2, kernel_size=(1, 1), stride=1)
    x = group_conv(x=x, filters=in_channel, kernel=kernel_size, stride=1, groups=4)

    # for shortcut
    # shortcut = DepthwiseConv2D_BN(shortcut, kernel_size, stride) # stride=2
    shortcut = Conv2D_BN_ReLU(shortcut, channel=out_channel - in_channel, kernel_size=(1, 1), stride=1)

    x = tf.concat([shortcut, x], axis=3)
    x = channel_shuffle(x, 2)
    return x
    '''
def channel_shuffle(x, groups):
    if K.image_data_format() == 'channels_last':
        height, width, in_channels = K.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, height, width, groups, channels_per_group]
        dim = (0, 1, 2, 4, 3)
        later_shape = [-1, height, width, in_channels]
    else:
        in_channels, height, width = K.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, groups, channels_per_group, height, width]
        dim = (0, 2, 1, 3, 4)
        later_shape = [-1, in_channels, height, width]
 
    x = tf.keras.layers.Lambda(lambda z: K.reshape(z, pre_shape))(x)
    x = tf.keras.layers.Lambda(lambda z: K.permute_dimensions(z, dim))(x)  
    x = tf.keras.layers.Lambda(lambda z: K.reshape(z, later_shape))(x)
 
    return x'''


MOBILENETV2_CONV_DEFS = [
    Conv(kernel=9, stride=1, depth=32, padding='same', activation=actv_str, bn=False),
    BottleNeck(kernel=7, stride=1, depth=32, padding='same', t_rate=4, n_rate=1),
    BottleNeck(kernel=7, stride=1, depth=48, padding='same', t_rate=4, n_rate=1),
    BottleNeck(kernel=7, stride=1, depth=48, padding='same', t_rate=4, n_rate=1),
    BottleNeck(kernel=5, stride=1, depth=64, padding='same', t_rate=4, n_rate=1),
    BottleNeck(kernel=5, stride=1, depth=64, padding='same', t_rate=4, n_rate=1),
    BottleNeck(kernel=5, stride=1, depth=128, padding='same', t_rate=4, n_rate=1),
    BottleNeck(kernel=5, stride=1, depth=128, padding='same', t_rate=4, n_rate=1),
]


def bottleneck(inputs, filters, kernel, t=4, strides=1, n=1):
    channel_axis = -1
    y = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding='same')(inputs)
    y = tf.keras.layers.BatchNormalization(axis=channel_axis)(y)
    y = tf.keras.layers.Activation(tf.nn.relu6)(y)  # 'linear'

    x = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=(1, 1), strides=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel, strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation('linear')(x)  #
    x = tf.keras.layers.add([x, y])
    return x


# ------------ mobilenet V2------------------#
def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)

    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=channel_axis)(x)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """
    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def _conv_block(x, filters, kernel=(1, 1), stride=(1, 1)):
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)
    return x


def create_s2p_model(input_window_length):
    actv = "relu"
    input_layer = tf.keras.layers.Input(shape=(input_window_length,))
    reshape_layer = tf.keras.layers.Reshape((1, input_window_length, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Convolution2D(filters=32, kernel_size=(10, 1), strides=(1, 1), padding="same",
                                                 activation=actv)(reshape_layer)
    conv_layer_2 = tf.keras.layers.Convolution2D(filters=32, kernel_size=(8, 1), strides=(1, 1), padding="same",
                                                 activation=actv)(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Convolution2D(filters=48, kernel_size=(6, 1), strides=(1, 1), padding="same",
                                                 activation=actv)(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Convolution2D(filters=64, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                                 activation=actv)(conv_layer_3)
    conv_layer_5 = tf.keras.layers.Convolution2D(filters=64, kernel_size=(5, 1), strides=(1, 1), padding="same",
                                                 activation=actv)(conv_layer_4)
    flatten_layer = tf.keras.layers.Flatten()(conv_layer_5)
    label_layer = tf.keras.layers.Dense(512, activation=actv)(flatten_layer)
    output_layer = tf.keras.layers.Dense(1, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def save_model(model, network_type, algorithm, appliance, save_model_dir):
    # model_path = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_path = save_model_dir

    if not os.path.exists(model_path):
        open((model_path), 'a').close()

    model.save(model_path)


def load_model(model, network_type, algorithm, appliance, saved_model_dir):
    # model_name = "saved_models/" + appliance + "_" + algorithm + "_" + network_type + "_model.h5"
    model_name = saved_model_dir
    print("PATH NAME: ", model_name)

    model = tf.keras.models.load_model(model_name, custom_objects={'mnloss': mnloss, 'relu6': tf.nn.relu6})  #
    num_of_weights = model.count_params()
    print("Loaded model with ", str(num_of_weights), " weights")
    return model


'''
def MobileNetv2(input_shape, k):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, layer repeat times.
    # Returns
        MobileNetv2 model.
    """
    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same')(x)

    x = Activation('softmax', name='softmax')(x)
    output = Reshape((k,))(x)

    model = Model(inputs, output)
    plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    return model
'''

'''
def depthwise_res_block(x, filters, kernel, stride, t, resdiual=False):
    input_tensor = x
    exp_channels = x.shape[-1]*t  #扩展维度
    x = conv_block(x, exp_channels, (1,1), (1,1))
    x = tf.keras.layers.DepthwiseConv2D(kernel, padding='same', strides=stride)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.nn.relu6)(x)
    x = tf.keras.layers.Conv2D(filters, (1,1), padding='same', strides=(1,1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if resdiual:
        x = tf.keras.layers.add([x, input_tensor])
    return x

def inverted_residual_layers(x, filters, stride, t, n):
    x = depthwise_res_block(x, filters, (3,1), stride, t, False)
    for i in range(1, n):
        x = depthwise_res_block(x, filters, (3,1), (1,1), t, True)
    return x


'''


def Conv2D_BN_ReLU(inputs, channel, kernel_size=(1, 1), stride=1, training=True):
    """Conv2D -> BN -> ReLU"""
    x = tf.keras.layers.Conv2D(channel, kernel_size, strides=stride,
                               padding="SAME", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)  # axis=-1, momentum=0.9, epsilon=1e-5
    x = tf.keras.layers.Activation("relu")(x)

    return x


def DepthwiseConv2D_BN(inputs, kernel_size=(3, 3), stride=1, training=True):
    """DepthwiseConv2D -> BN"""
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=stride,
                                        depth_multiplier=1,
                                        padding="SAME", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)  # axis=-1, momentum=0.9, epsilon=1e-5
    x = tf.keras.layers.Activation("relu")(x)
    return x


def ShufflenetUnit1(inputs, out_channel, kernel_size=(1, 5), training=False):
    """The unit of shufflenetv2 for stride=1
    Args:
        out_channel: int, number of channels
    """
    assert out_channel % 2 == 0
    shortcut, x = tf.split(inputs, 2, axis=3)
    x = Conv2D_BN_ReLU(x, channel=out_channel // 2, kernel_size=(1, 1), stride=1)
    x = DepthwiseConv2D_BN(x, kernel_size, stride=1)
    x = Conv2D_BN_ReLU(x, channel=out_channel // 2, kernel_size=(1, 1), stride=1)

    x = tf.concat([shortcut, x], axis=3)
    x = channel_shuffle(x, 2)
    return x


def ShufflenetUnit2(inputs, in_channel, out_channel, kernel_size=(1, 5), stride=1, training=False):
    """The unit of shufflenetv2 for stride=2"""

    assert out_channel % 2 == 0
    shortcut, x = inputs, inputs
    x = Conv2D_BN_ReLU(x, channel=out_channel // 2, kernel_size=(1, 1), stride=1)
    x = DepthwiseConv2D_BN(x, kernel_size, stride)  # stride=2
    x = Conv2D_BN_ReLU(x, channel=out_channel - in_channel, kernel_size=(1, 1), stride=1)

    # for shortcut
    shortcut = DepthwiseConv2D_BN(shortcut, kernel_size, stride)  # stride=2
    shortcut = Conv2D_BN_ReLU(shortcut, channel=in_channel, kernel_size=(1, 1), stride=1)

    x = tf.concat([shortcut, x], axis=3)
    x = channel_shuffle(x, 2)
    return x


class ShufflenetStage(tf.keras.Model):
    """The stage of shufflenet"""

    def __init__(self, in_channel, out_channel, num_blocks):
        super(ShufflenetStage, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.ops = []
        for i in range(num_blocks):
            if i == 0:
                op = ShufflenetUnit2(in_channel, out_channel)
            else:
                op = ShufflenetUnit1(out_channel)
            self.ops.append(op)

    def call(self, inputs, training=False):
        x = inputs
        for op in self.ops:
            x = op(x, training=training)
        return x
