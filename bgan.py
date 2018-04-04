from __future__ import division, print_function
import numpy as np
import scipy
import scipy.io as sio
import cv2
from generator import Vgg19
import tensorflow as tf
try:
    import cPickle
except:
    import pickle as cPickle
# from pydatset import get_CIFAR10_data
import matplotlib.pyplot as plt

#parameters
data_batch_size = 10000
batch_size = 1
epochs = 1
lr = 0.0001
w64 = 64
h64 = 64
w224 = 224
h224 = 224
channels = 3
S = sio.loadmat('./S_K1_20_K2_30.mat')['S']  # similarity matrix

#placeholders
true_img_64 = tf.placeholder(tf.float32, [batch_size, w64,h64,channels])
true_img_224 = tf.placeholder(tf.float32, [batch_size, w224,h224,channels])
beta_nima = tf.placeholder(tf.float32,[1])

print("X64 placeholder", tf.shape(true_img_64))
print("X224 placeholder", tf.shape(true_img_224))

# Y = tf.placeholder(tf.float32, [batch_size, channels, w, h])

def read_data():

    f = open('datasets/cifar-10-batches-py/data_batch_1', 'rb')
    try:
        datadict = cPickle.load(f)
    except:
        f.seek(0)
        # datadict = cPickle.load(f, encoding="utf-8")
        datadict = cPickle.load(f, encoding="latin1")
    f.close()
    features = datadict["data"]
    # labels = datadict['labels']
    features = features.reshape(data_batch_size, channels, 32, 32).transpose(0,2,3,1).astype("uint8")
    #change resize to 64*64
    # labels = np.array(labels)
    print("while reading data - features.shape", features.shape)
    x224 = []
    x64 = []
    # for _, image_file in enumerate(features):
    for _, img in enumerate(features):
       # img = cv2.imread(image_file)
       img224 = cv2.resize(img, (w224, h224)).astype(np.float32)    # i = np.random.choice(range(len(X)))
       x224.append(img224)
       img64 = cv2.resize(img, (w64, h64)).astype(np.float32)    # i = np.random.choice(range(len(X)))
       x64.append(img64)
    # print("len(images)=", len(images))
    # print("images[0].shape=", images[0].shape)
    # # plt.imsave('h1.png',X[i:i+1][0])

    return x224, x64

def Encoder(inputs): # change use 224 shape
    with tf.variable_scope("enc", reuse=False) as scope:

        c0 = tf.layers.conv2d(inputs = inputs,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")

        c1 = tf.layers.conv2d(inputs = c0,filters=64,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")

        mp0 = tf.layers.max_pooling2d(inputs = c1,pool_size = 2,strides = 2,  padding='same',name="mp0")



        c2 = tf.layers.conv2d(inputs = mp0,filters=128,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")

        c3 = tf.layers.conv2d(inputs = c2,filters=128,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

        mp1 = tf.layers.max_pooling2d(inputs = c3,pool_size = 2,strides = 2,  padding='same',name="mp1")



        c4 = tf.layers.conv2d(inputs = mp1,filters=256,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv4")

        c5 = tf.layers.conv2d(inputs = c4,filters=256,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv5")

        c6 = tf.layers.conv2d(inputs = c5,filters=256,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv6")

        c6 = tf.layers.conv2d(inputs = c6,filters=256,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv6_extra")

        # change add conv layer

        mp2 = tf.layers.max_pooling2d(inputs = c6,pool_size = 2,strides = 2,  padding='same',name="mp2")



        c7 = tf.layers.conv2d(inputs = mp2,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv7")

        c8 = tf.layers.conv2d(inputs = c7,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv8")

        c9 = tf.layers.conv2d(inputs = c8,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv9")

        c9 = tf.layers.conv2d(inputs = c9,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv9_extra")

        #conv layer
        mp3 = tf.layers.max_pooling2d(inputs = c9,pool_size = 2,strides = 2,  padding='same',name="mp3")



        c10 = tf.layers.conv2d(inputs = mp3,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv10")

        c11 = tf.layers.conv2d(inputs = c10,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv11")

        c12 = tf.layers.conv2d(inputs = c11,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv12")

        c12 = tf.layers.conv2d(inputs = c12,filters=512,kernel_size=3,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv12_extra")

        #conv layer
        mp4 = tf.layers.max_pooling2d(inputs = c12,pool_size = 2,strides = 2,  padding='same',name="mp4")


        fc0 = tf.contrib.layers.fully_connected(mp4, 4096,activation_fn=tf.nn.relu)
        fc1 = tf.contrib.layers.fully_connected(fc0, 4096,activation_fn=tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 32)
        t0 = tf.nn.tanh(fc2)


        #change save fully_connected

        return t0, fc2


def hash_layer(inputs,beta=1.0,approximation="tanh"):# dimensions
    if approximation == "tanh":
        t = tf.nn.tanh(beta*inputs)
        return t
    # if approximation == "app":
    #     app_bottom = np.zeros(len(inputs),dtype=np.int8) - 1  # [-1, ...]
    #     app_top = np.zeros(len(inputs),dtype=np.int8) + 1  # [+1, ...]
    #     return np.maximum(app_bottom, np.minimum(beta*inputs, app_top))
    # raise NotImplementedError("The approximation `{}` does not exist.".format(approximation))

def test_hash_layer():
    test_range1 = range(-20, 20, 1)
    test_layer1 = np.array([v / 10 for v in test_range1])
    test_tanh_hash1 = hash_layer(test_layer1)
    test_app_hash1 = hash_layer(test_layer1,approximation="app")
    print(test_layer1)
    print(test_tanh_hash1)
    print(test_app_hash1)

# test_hash_layer()

def generator(inputs):#change check shape #change range of true and gen images
    with tf.variable_scope("gen", reuse=False) as scope:
        fc0 = tf.contrib.layers.fully_connected(inputs, 16384)
        fc0 = tf.reshape(fc0, [batch_size, 8, 8, 256])
        c0 = tf.layers.conv2d_transpose(inputs = fc0,filters=256,kernel_size=5,activation = tf.nn.relu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")
        c1 = tf.layers.conv2d_transpose(inputs = c0,filters=128,kernel_size=5,activation = tf.nn.relu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")
        c2 = tf.layers.conv2d_transpose(inputs = c1,filters=32,kernel_size=5,activation = tf.nn.relu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")
        c3 = tf.layers.conv2d_transpose(inputs = c2,filters=3,kernel_size=1,activation = tf.nn.relu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

        b0 = tf.layers.batch_normalization(inputs = c3,name = "b0")#change do not need?
        r0 = tf.nn.elu(b0, name = "r0") #change or sigmoid?
        return r0

def discriminator(inputs,reuse=False):

    with tf.variable_scope("disc", reuse=reuse) as scope:
        c0 = tf.layers.conv2d(inputs = inputs,filters=32,kernel_size=5,activation = tf.nn.elu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")

        c1 = tf.layers.conv2d(inputs = c0,filters=128,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")

        c2 = tf.layers.conv2d(inputs = c1,filters=256,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")

        c3 = tf.layers.conv2d(inputs = c2,filters=512,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

        c3 = tf.contrib.layers.flatten(c3)

        fc0 = tf.contrib.layers.fully_connected(c3, 1024)
        s0 = tf.nn.sigmoid(fc0, name = "s0")

        return c3,s0

def N_losses(b,s):

    #neighborhood loss
    mm = tf.matmul(b,b,transpose_b = True)
    # foo = (1/tf.size(b, out_type=tf.float32) * mm)
    foo = (1/32 * mm)
    asum = tf.square(foo - s )
    N_loss = 0.5 * tf.reduce_sum(asum)
    # N_loss = 0.5 * tf.reduce_sum(tf.square((1/tf.size(b) * tf.matmul(tf.transpose(b),b)) - s ))
    return N_loss

def C_losses(true_img,gen_img):
    #content loss
    last_conv_true_img,_ = discriminator(true_img, True)
    last_conv_gen_img,_ = discriminator(gen_img, True)

    #check shapes
    MSE_loss = tf.reduce_mean(tf.square(true_img-gen_img))
    P_loss = tf.reduce_mean(tf.square(last_conv_true_img - last_conv_gen_img))

    C_loss = MSE_loss + P_loss
    return C_loss

def A_losses(true_img,gen_img):
    #adversarial loss
    _,true_out = discriminator(true_img,True) # save output value in C_loss
    _,gen_out = discriminator(gen_img,True)
    A_loss = tf.log(true_out) + tf.log(1-gen_out)

    return A_loss


# encoder_output = Encoder(true_img_224)
train_model = tf.placeholder(tf.bool)
with tf.variable_scope("enc"):
    vgg_net = Vgg19('./vgg19.npy', codelen=32)
    vgg_net.build(true_img_224, beta_nima, train_model)
    z_x_mean = vgg_net.fc9
    z_x_log_sigma_sq = vgg_net.fc10
print("encoder_output.shape=", z_x_mean.shape)
b = hash_layer(z_x_mean)
print("b.shape=", b.shape)
gen_img = generator(b)
disc_image = discriminator(gen_img)

t_vars = tf.trainable_variables()
print("t_vars=", t_vars)

e_vars = [var for var in t_vars if "enc" in var.name]
g_vars = [var for var in t_vars if "gen" in var.name]
d_vars = [var for var in t_vars if "disc" in var.name]

n_l = N_losses(b,S)
c_l = C_losses(true_img_64,gen_img)
a_l = A_losses(true_img_64,gen_img)

e_loss = n_l + c_l
g_loss = c_l + a_l
d_loss = a_l

e_optim = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(e_loss, var_list=e_vars)
g_optim = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(g_loss, var_list=g_vars)
d_optim = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(d_loss, var_list=d_vars)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    features224, features64 = read_data()
    # print features.shape

    for e in range(epochs):
        for i in range(0, len(features224), batch_size):

            # f_64 = features[i].reshape(batch_size,w64,w64,channels)
            f_224 = features224[i:i+batch_size]
            f_64 = features64[i:i+batch_size]
            # optimizer = sess.run(e_optim,feed_dict = {true_img_64: f_64, true_img_224: })
            optimizer = sess.run(e_optim,feed_dict = {true_img_64: f_64, true_img_224: f_224, beta_nima:[-2], train_model: True})

            for g_step in range(5):
                g_optimizer = sess.run(g_optim,feed_dict = {true_img_64: f_64})
            for d_step in range(1):
                d_optimizer = sess.run(d_optim,feed_dict = {true_img_64: f_64})
