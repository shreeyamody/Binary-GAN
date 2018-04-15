from __future__ import division, print_function
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np
import scipy
import scipy.io as sio
import cv2
from generator import Vgg19
try:
    import cPickle
except:
    import pickle as cPickle
# from pydatset import get_CIFAR10_data
import matplotlib
import matplotlib.pyplot as plt

#parameters
data_batch_size = 10000
batch_size = 40
epochs = 1
lr = 0.001
w64 = 64
h64 = 64
w224 = 224
h224 = 224
channels = 3
# S = sio.loadmat('./S_K1_20_K2_30.mat')['S']  # similarity matrix
S = np.load('S.npz')['arr_0']#cifar-10
# S = np.load('S_20_30_celeba_KNN_1_4.npz')['arr_0']

#placeholders
true_img_64 = tf.placeholder(tf.float32, [batch_size, w64,h64,channels])
true_img_224 = tf.placeholder(tf.float32, [batch_size, w224,h224,channels])
beta_nima = tf.placeholder(tf.float32,[1])
train_model = tf.placeholder(tf.bool)
s = tf.placeholder(tf.float32, [batch_size, batch_size])

def data_iterator(img224):
    while True:
        idxs = np.arange(0, len(img224))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(img224), batch_size):
            cur_idxs = idxs[batch_idx:batch_idx + batch_size]
            images_batch = img224[cur_idxs]
            if len(images_batch) < batch_size:
                break
            images_batch = images_batch.astype("float32")
            yield images_batch ,cur_idxs

def read_data():

    f = open('datasets/cifar-10-batches-py/data_batch_1', 'rb') #cifar-10
    # f = open('datasets/img_align_celeba','rb')
    try:
        datadict = cPickle.load(f)
    except:
        f.seek(0)
        # datadict = cPickle.load(f, encoding="utf-8")
        datadict = cPickle.load(f, encoding="latin1")
    f.close()
    features = datadict["data"]
    features = features.reshape(data_batch_size, channels, 32, 32).transpose(0,2,3,1).astype("uint8")
    #change resize to 64*64

    # print("while reading data - features.shape", features.shape)
    x224 = []
    x64 = []
    # for _, image_file in enumerate(features):
    for _, img in enumerate(features):
       # img = cv2.imread(image_file)
       img224 = cv2.resize(img, (w224, h224)).astype(np.float32)    # i = np.random.choice(range(len(X)))
       x224.append(img224)
       img64 = cv2.resize(img, (w64, h64)).astype(np.float32)    # i = np.random.choice(range(len(X)))
       x64.append(img64/255.0) # to bring it in the range of [0,1]
    # print("len(images)=", len(images))
    # print("images[0].shape=", images[0].shape)
    # # plt.imsave('h1.png',X[i:i+1][0])

    return x224, x64

# def Encoder(inputs): # change use 224 shape
#     with tf.variable_scope("enc", reuse=False) as scope:
#
#         c0 = tf.layers.conv2d(inputs = inputs,filters=64,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")
#
#         c1 = tf.layers.conv2d(inputs = c0,filters=64,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")
#
#         mp0 = tf.layers.max_pooling2d(inputs = c1,pool_size = 2,strides = 2,  padding='same',name="mp0")
#
#
#
#         c2 = tf.layers.conv2d(inputs = mp0,filters=128,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")
#
#         c3 = tf.layers.conv2d(inputs = c2,filters=128,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")
#
#         mp1 = tf.layers.max_pooling2d(inputs = c3,pool_size = 2,strides = 2,  padding='same',name="mp1")
#
#
#
#         c4 = tf.layers.conv2d(inputs = mp1,filters=256,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv4")
#
#         c5 = tf.layers.conv2d(inputs = c4,filters=256,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv5")
#
#         c6 = tf.layers.conv2d(inputs = c5,filters=256,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv6")
#
#         c6 = tf.layers.conv2d(inputs = c6,filters=256,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv6_extra")
#
#         # change add conv layer
#
#         mp2 = tf.layers.max_pooling2d(inputs = c6,pool_size = 2,strides = 2,  padding='same',name="mp2")
#
#
#
#         c7 = tf.layers.conv2d(inputs = mp2,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv7")
#
#         c8 = tf.layers.conv2d(inputs = c7,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv8")
#
#         c9 = tf.layers.conv2d(inputs = c8,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv9")
#
#         c9 = tf.layers.conv2d(inputs = c9,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv9_extra")
#
#         #conv layer
#         mp3 = tf.layers.max_pooling2d(inputs = c9,pool_size = 2,strides = 2,  padding='same',name="mp3")
#
#
#
#         c10 = tf.layers.conv2d(inputs = mp3,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv10")
#
#         c11 = tf.layers.conv2d(inputs = c10,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv11")
#
#         c12 = tf.layers.conv2d(inputs = c11,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv12")
#
#         c12 = tf.layers.conv2d(inputs = c12,filters=512,kernel_size=3,activation = tf.nn.elu, strides=(1,1), \
#         kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv12_extra")
#
#         #conv layer
#         mp4 = tf.layers.max_pooling2d(inputs = c12,pool_size = 2,strides = 2,  padding='same',name="mp4")
#
#
#         fc0 = tf.contrib.layers.fully_connected(mp4, 4096,activation_fn=tf.nn.elu)
#         fc1 = tf.contrib.layers.fully_connected(fc0, 4096,activation_fn=tf.nn.elu)
#         fc2 = tf.contrib.layers.fully_connected(fc1, 32)
#         t0 = tf.nn.tanh(fc2)
#
#
#         #change save fully_connected
#
#         return t0, fc2

def hash_layer_new(z_x_mean,z_x_log_sigma_sq):
    eps = tf.random_normal((batch_size, 32), 0, 1) # normal dist for VAE
    z_x = tf.add(z_x_mean,tf.multiply(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps)) # grab our actual z
    return z_x

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

def generator(inputs,reuse=False):#change check shape #change range of true and gen images
    # inputs = tf.Print(inputs, [inputs], message="START `generator`:")
    with tf.variable_scope("gen",reuse=reuse) as scope:
        fc0 = tf.contrib.layers.fully_connected(inputs, 16384)
        fc0 = tf.reshape(fc0, [batch_size, 8, 8, 256])
        c0 = tf.layers.conv2d_transpose(inputs = fc0,filters=256,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")
        c1 = tf.layers.conv2d_transpose(inputs = c0,filters=128,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")
        c2 = tf.layers.conv2d_transpose(inputs = c1,filters=32,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")
        c3 = tf.layers.conv2d_transpose(inputs = c2,filters=3,kernel_size=1,activation = tf.sigmoid, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

        # b0 = tf.layers.batch_normalization(inputs = c3,name = "b0")#change do not need?
        # r0 = tf.nn.elu(b0, name = "r0") #change or sigmoid?
        return c3

def discriminator(inputs,reuse=False):
    # tf.Print("START `discriminator`:", inputs.shape)
    # inputs = tf.Print(inputs, [inputs], message="START `discriminator`:")

    with tf.variable_scope("disc", reuse=reuse) as scope:
        c0 = tf.layers.conv2d(inputs = inputs,filters=32,kernel_size=5,activation = tf.nn.elu, strides=(1,1), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv0")

        c1 = tf.layers.conv2d(inputs = c0,filters=128,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv1")

        c2 = tf.layers.conv2d(inputs = c1,filters=256,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv2")

        c3 = tf.layers.conv2d(inputs = c2,filters=256,kernel_size=5,activation = tf.nn.elu, strides=(2,2), \
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),padding = "SAME",name = "conv3")

        c3 = tf.contrib.layers.flatten(c3)

        fc0 = tf.contrib.layers.fully_connected(c3, 1024, activation_fn=tf.nn.elu)
        fc1 = tf.contrib.layers.fully_connected(fc0, 1, activation_fn=tf.nn.sigmoid)
        # s0 = tf.nn.sigmoid(fc0, name = "s0")

        return fc0,fc1

def N_losses(b,s):

    #neighborhood loss

    N_loss = 0.5 * (tf.reduce_mean(tf.square(tf.matmul(b,b,transpose_b = True) - s )) +\
     tf.reduce_mean(tf.square(b - tf.sign(b))))
    # N_loss = 0.5 * tf.reduce_sum(tf.square((1/tf.size(b) * tf.matmul(tf.transpose(b),b)) - s ))
    # N_loss = tf.Print(N_loss, [N_loss], message="N_loss:")

    return 10*N_loss

def C_losses(true_img,gen_img,last_conv_true_img,last_conv_gen_img):
    #content loss
    # last_conv_true_img,_ = discriminator(true_img, True)
    # last_conv_gen_img,_ = discriminator(gen_img, True)

    #check shapes
    #MSE_loss = tf.reduce_mean(tf.square(true_img-gen_img)) # diff
    P_loss = tf.reduce_sum(tf.square(last_conv_true_img - last_conv_gen_img))/64.0/64.0/3.0

    C_loss = P_loss # + MSE_loss
    # C_loss = tf.Print(C_loss, [C_loss], message="C_loss:")

    return C_loss

def D_losses(disc_true_image,disc_rand_gen_img):
    #adversarial loss
    # _,true_out = discriminator(true_img,True) # save output value in C_loss
    # _,gen_out = discriminator(gen_img,True)
    D_loss = tf.reduce_mean(-1. * (tf.log(tf.clip_by_value(disc_true_image,1e-5,1.0)) + tf.log(tf.clip_by_value(1-disc_rand_gen_img,1e-5,1.0))))
    # D_loss = tf.Print(D_loss, [D_loss], message="D_loss:")

    return D_loss



def G_losses(disc_rand_gen_img):
    #adversarial loss
    # _,true_out = discriminator(true_img,True) # save output value in C_loss
    # _,gen_out = discriminator(gen_img,True)
    G_loss = tf.reduce_mean(-1. * tf.log(tf.clip_by_value(disc_rand_gen_img,1e-5,1.0)))
    # G_loss = tf.Print(G_loss, [G_loss], message="G_loss:")

    return G_loss

# encoder_output = Encoder(true_img_224)
rand_z = tf.random_normal((batch_size, 32), 0, 1)
with tf.variable_scope("enc") as scope:
    vgg_net = Vgg19('./vgg19.npy', codelen=32)
    vgg_net.build(true_img_224, beta_nima, train_model)
    z_x_mean = vgg_net.fc9
    z_x_log_sigma_sq = vgg_net.fc10

# print("encoder_output.shape=", z_x_mean.shape)

# print("b.shape=", b.shape)
with tf.variable_scope("gen") as scope:
    b = hash_layer_new(z_x_mean,z_x_log_sigma_sq)
    gen_img = generator(b)
    rand_gen_img = generator(rand_z,True)

with tf.variable_scope("disc") as scope:
    last_conv_gen_img, disc_gen_image = discriminator(gen_img)
    last_conv_true_img, disc_true_image = discriminator(true_img_64,True)
    last_conv_rand_gen_img, disc_rand_gen_img = discriminator(rand_gen_img,True)

t_vars = tf.trainable_variables()
# print("t_vars=", t_vars)

e_vars = [var for var in t_vars if "enc" in var.name]
g_vars = [var for var in t_vars if "gen" in var.name]
d_vars = [var for var in t_vars if "disc" in var.name]

n_l = N_losses(b,s)
c_l = C_losses(true_img_64, gen_img,last_conv_true_img, last_conv_gen_img)
g_l = G_losses(disc_rand_gen_img)
d_l = D_losses(disc_true_image, disc_rand_gen_img)

e_loss = (n_l + c_l)
g_loss = (c_l + g_l)
d_loss = d_l

e_optim = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(e_loss, var_list=e_vars)
g_optim = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(g_loss, var_list=g_vars)
d_optim = tf.train.AdamOptimizer(0.0001, beta1 = 0.0, beta2 = 0.9).minimize(d_loss, var_list=d_vars)

saver = tf.train.Saver()

# with tf.InteractiveSession() as sess:
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    features224, features64 = read_data()
    features224=np.array(features224)
    features64=np.array(features64)
    num_examples = len(features64)
    total_batch = int(np.floor(num_examples / batch_size ))

    # if False:
    for e in range(epochs):
        print("Begin epoch ", e)
        iter_ = data_iterator(features224)
        for i in range(total_batch):
            next_batches224 ,indx3= iter_.next()
            next_batches64 = features64[indx3]
            ss = S[indx3,:][:,indx3]
            print("Using images ", i, " to ", i + batch_size)
            # f_64 = features[i].reshape(batch_size,w64,w64,channels)
            # f_224 = features224[i:i+batch_size]
            # f_64 = features64[i:i+batch_size]
            # optimizer = sess.run(e_optim,feed_dict = {true_img_64: f_64, true_img_224: })
            # print("Begin optimizing e.")
            e_optimizer = sess.run(e_optim,feed_dict = {true_img_64: next_batches64, true_img_224: next_batches224, beta_nima:[-2], \
            train_model: True, s:ss})
            # print("Begin optimizing g.")
            for g_step in range(1):
                g_img,g_optimizer = sess.run([gen_img,g_optim],feed_dict = {true_img_64: next_batches64, true_img_224: next_batches224,\
                 beta_nima:[-2], train_model: True, s:ss})
                # g_img = np.reshape(g_img,[64,64,3])
                for t in range(batch_size):
                    matplotlib.image.imsave('gen6/g_img_{}_{}_{}t.png'.format(e,i,t),g_img[t])
            # print("Begin optimizing d.")
            for d_step in range(1):
                d_optimizer = sess.run(d_optim,feed_dict = {true_img_64: next_batches64, true_img_224: next_batches224, beta_nima:[-2], \
                train_model: True, s:ss})


    save_path = saver.save(sess, "model.ckpt")
    # else:
    # test images
    test_dataset = sio.loadmat('cifar-10.mat')['test_data']  #cifar-10 data
    test_images224 = []
    test_images64 = []
    print ("starting test")
    for i in range(len(test_dataset)):
        print ("starting test",i)

        t = test_dataset[:, :, :, i]
        image224 = scipy.misc.imresize(t, [224, 224])
        image64 = scipy.misc.imresize(t, [64, 64])
        test_images224.append(image224)
        test_images64.append(image64)
    print ("Done for loop")

    test_images224 = np.array(test_images224)
    test_images64 = np.array(test_images64)
    print ("restoring")

    restore = saver.restore(sess, "model.ckpt")
    # restore_vars = chkp.print_tensors_in_checkpoint_file("model.ckpt", tensor_name='', all_tensors=True)
    test_iter = data_iterator(test_images224)
    test_total_batch = int(np.floor(len(test_images224) / batch_size))
    for i in range(test_total_batch):
        print ("test_total_batch",i)

        test_next_batches224, idx4 = test_iter.next()
        test_next_batches64 = test_images64[idx4]

        # true_img_64 = graph.get_tensor_by_name("true_img_64:0")
        # true_img_224 = graph.get_tensor_by_name("true_img_224:0")
        # beta_nima = graph.get_tensor_by_name("beta_nima:0")
        # train_model = graph.get_tensor_by_name("train_model:0")
        print ("starting sess")

        g,rg = sess.run([gen_img,rand_gen_img], feed_dict={true_img_64: test_next_batches64,true_img_224: test_next_batches224,beta_nima:[-2],\
         train_model: False}) #change to test images!!!!!!!!!!
        print ("saving pics")

        for k in range(batch_size):
            matplotlib.image.imsave('gen5/true_img_{}.png'.format(k),test_next_batches64[k])
            matplotlib.image.imsave('gen5/test_gen_img_{}.png'.format(k),g[k])
            matplotlib.image.imsave('gen5/test_rand_gen_img.png'.format(k),rg[k])
