import numpy as np
import tensorflow as tf
import cPickle
# from pydatset import get_CIFAR10_data
import matplotlib.pyplot as plt

#parameters
batch_size = 1
epochs = 10
lr = 0.0001
w = 32
h = 32

#placeholders
X = tf.placeholder(tf.float32, [batch_size, channel, w, h])
# Y = tf.placeholder(tf.float32, [batch_size, channel, w, h])

def read_data():

    f = open('cifar-10-batches-py/data_batch_1', 'rb')
    datadict = cPickle.load(f)
    f.close()
    X = datadict["data"]
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)
    # i = np.random.choice(range(len(X)))
    # plt.imsave('h1.png',X[i:i+1][0])

    return X,Y


def Encoder(inputs):
    c0 = tf.layers.conv2d(inputs = inputs,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")

    c1 = tf.layers.conv2d(inputs = c0,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")

    mp0 = tf.layers.max_pooling2d(inputs = c1,pool_size = 2,strides = 2,padding='same',name="mp0")



    c2 = tf.layers.conv2d(inputs = mp0,filters=128,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")

    c3 = tf.layers.conv2d(inputs = c2,filters=128,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

    mp1 = tf.layers.max_pooling2d(inputs = c3,pool_size = 2,strides = 2,padding='same',name="mp1")



    c4 = tf.layers.conv2d(inputs = mp1,filters=256,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv4")

    c5 = tf.layers.conv2d(inputs = c4,filters=256,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv5")

    c6 = tf.layers.conv2d(inputs = c45,filters=256,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv6")

    mp2 = tf.layers.max_pooling2d(inputs = c6,pool_size = 2,strides = 2,padding='same',name="mp2")



    c7 = tf.layers.conv2d(inputs = mp2,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv7")

    c8 = tf.layers.conv2d(inputs = c7,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv8")

    c9 = tf.layers.conv2d(inputs = c8,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv9")

    mp3 = tf.layers.max_pooling2d(inputs = c9,pool_size = 2,strides = 2,padding='same',name="mp3")



    c10 = tf.layers.conv2d(inputs = mp3,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv10")

    c11 = tf.layers.conv2d(inputs = c10,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv11")

    c12 = tf.layers.conv2d(inputs = c11,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv12")

    mp4 = tf.layers.max_pooling2d(inputs = c12,pool_size = 2,strides = 2,padding='same',name="mp4")



    fc0 = tf.contrib.layers.fully_connected(mp4, 4096, name = "fc0")
    fc1 = tf.contrib.layers.fully_connected(fc0, 4096, name = "fc1")

    return fc1


def hash_layer(inputs,beta=1.0,approximation="tanh"):
    if approximation == "tanh":
        return np.tanh(beta*inputs)
    if approximation == "app":
        app_bottom = np.zeros(len(inputs),dtype=np.int8) - 1  # [-1, ...]
        app_top = np.zeros(len(inputs),dtype=np.int8) + 1  # [+1, ...]
        return np.maximum(app_bottom, np.minimum(beta*inputs, app_top))
    raise NotImplementedError("The approximation `{}` does not exist.".format(approximation))

def test_hash_layer():
    test_range1 = range(-20, 20, 1)
    test_layer1 = np.array([v / 10 for v in test_range1])
    test_tanh_hash1 = hash_layer(test_layer1)
    test_app_hash1 = hash_layer(test_layer1,approximation="app")
    print(test_layer1)
    print(test_tanh_hash1)
    print(test_app_hash1)

# test_hash_layer()

def generator(inputs):

    fc0 = tf.contrib.layers.fully_connected(inputs, 16384, name = "fc0")
    c0 = tf.layers.conv2d_transpose(inputs = fc0,filters=256,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")
    c1 = tf.layers.conv2d_transpose(inputs = c0,filters=128,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")
    c2 = tf.layers.conv2d_transpose(inputs = c1,filters=32,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")
    c3 = tf.layers.conv2d_transpose(inputs = c2,filters=3,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

    b0 = tf.layers.batch_normalization(inputs = c3,name = "b0")
    r0 = tf.nn.relu(inputs = b0, name = "r0")
    return r0

def discriminator(inputs):

    c0 = tf.layers.conv2d(inputs = inputs,filters=32,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")

    c1 = tf.layers.conv2d(inputs = c0,filters=128,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")

    c2 = tf.layers.conv2d(inputs = c1,filters=256,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")

    c3 = tf.layers.conv2d(inputs = c2,filters=512,kernel_size=5,activation = tf.nn.relu, strides=(1,1), \
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

    fc0 = tf.contrib.layers.fully_connected(inputs, 1024, name = "fc0")
    s0 = tf.nn.sigmoid(inputs = fc0, name = "s0")

    return s0

encoder_ouput = Encoder(X)
hash_output = hash_layer(encoder_output)
generator_output = generator(hash_output)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    features,labels = read_data()

    for e in range(epochs):
        num_batches = int(len(features)/batch_size)
        optimizer = sess.run(generator_output,feed_dict = {X: features, Y: labels})
