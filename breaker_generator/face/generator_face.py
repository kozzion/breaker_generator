import os
import time

#from glob import glob

import random

from PIL import Image

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class GeneratorFace:

    def __init__(self) -> None:
            # Hyperparaeters
        self.IMAGE_SIZE = 128
        self.NOISE_SIZE = 100
        self.LR_D = 0.00004
        self.LR_G = 0.0002
        self.BATCH_SIZE = 64
        self.EPOCH = 0 # Non-zero only if we are resuming training with model checkpoint
        self.EPOCHS = 5 #EPOCH + number of epochs to perform
        self.BETA1 = 0.5
        self.WEIGHT_INIT_STDDEV = 0.02
        self.EPSILON = 0.00005
        self.SAMPLES_TO_SHOW = 5


        # Data (https://www.kaggle.com/greg115/celebrities-100k)
        self.BASE_PATH = "C:\\project\\data\\data_common\\celebrity_100k\\"
        self.DATASET_LIST_PATH = self.BASE_PATH + "100k.txt"
        self.INPUT_DATA_DIR = self.BASE_PATH + "100k/100k/"
        self.OUTPUT_DIR = "./"
        self.DATASET = [self.INPUT_DATA_DIR + str(line).rstrip() for line in open(self.DATASET_LIST_PATH,"r")]
        self.DATASET_SIZE = len(self.DATASET) 
        self.MINIBATCH_SIZE = self.DATASET_SIZE // self.BATCH_SIZE

        # Optional - model path to resume training
        #MODEL_PATH = BASE_PATH + "models/" + "model_" + str(EPOCH) + ".ckpt"m
        
    def generator(self, z, output_channel_dim, training):
        with tf.compat.v1.variable_scope("generator", reuse= not training):
            
            # 8x8x1024
            fully_connected = tf.compat.v1.layers.dense(z, 8*8*1024)
            fully_connected = tf.reshape(fully_connected, (-1, 8, 8, 1024))
            fully_connected = tf.nn.leaky_relu(fully_connected)

            # 8x8x1024 -> 16x16x512
            trans_conv1 = tf.compat.v1.layers.conv2d_transpose(inputs=fully_connected,
                                                    filters=512,
                                                    kernel_size=[5,5],
                                                    strides=[2,2],
                                                    padding="SAME",
                                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                                    name="trans_conv1")
            batch_trans_conv1 = tf.compat.v1.layers.batch_normalization(inputs = trans_conv1,
                                                            training=training,
                                                            epsilon=self.EPSILON,
                                                            name="batch_trans_conv1")
            trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1,
                                            name="trans_conv1_out")
            
            # 16x16x512 -> 32x32x256
            trans_conv2 = tf.compat.v1.layers.conv2d_transpose(inputs=trans_conv1_out,
                                                    filters=256,
                                                    kernel_size=[5,5],
                                                    strides=[2,2],
                                                    padding="SAME",
                                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                                    name="trans_conv2")
            batch_trans_conv2 = tf.compat.v1.layers.batch_normalization(inputs = trans_conv2,
                                                            training=training,
                                                            epsilon=self.EPSILON,
                                                            name="batch_trans_conv2")
            trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2,
                                            name="trans_conv2_out")
            
            # 32x32x256 -> 64x64x128
            trans_conv3 = tf.compat.v1.layers.conv2d_transpose(inputs=trans_conv2_out,
                                                    filters=128,
                                                    kernel_size=[5,5],
                                                    strides=[2,2],
                                                    padding="SAME",
                                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                                    name="trans_conv3")
            batch_trans_conv3 = tf.compat.v1.layers.batch_normalization(inputs = trans_conv3,
                                                            training=training,
                                                            epsilon=self.EPSILON,
                                                            name="batch_trans_conv3")
            trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3,
                                            name="trans_conv3_out")
            
            # 64x64x128 -> 128x128x64
            trans_conv4 = tf.compat.v1.layers.conv2d_transpose(inputs=trans_conv3_out,
                                                    filters=64,
                                                    kernel_size=[5,5],
                                                    strides=[2,2],
                                                    padding="SAME",
                                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                                    name="trans_conv4")
            batch_trans_conv4 = tf.compat.v1.layers.batch_normalization(inputs = trans_conv4,
                                                            training=training,
                                                            epsilon=self.EPSILON,
                                                            name="batch_trans_conv4")
            trans_conv4_out = tf.nn.leaky_relu(batch_trans_conv4,
                                            name="trans_conv4_out")
            
            # 128x128x64 -> 128x128x3
            logits = tf.compat.v1.layers.conv2d_transpose(inputs=trans_conv4_out,
                                                filters=3,
                                                kernel_size=[5,5],
                                                strides=[1,1],
                                                padding="SAME",
                                                kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                                name="logits")
            out = tf.tanh(logits, name="out")
            return out

    def discriminator(self, x, reuse):
        with tf.compat.v1.variable_scope("discriminator", reuse=reuse): 
            
            # 128*128*3 -> 64x64x64 
            conv1 = tf.compat.v1.layers.conv2d(inputs=x,
                                    filters=64,
                                    kernel_size=[5,5],
                                    strides=[2,2],
                                    padding="SAME",
                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                    name='conv1')
            batch_norm1 = tf.compat.v1.layers.batch_normalization(conv1,
                                                        training=True,
                                                        epsilon=self.EPSILON,
                                                        name='batch_norm1')
            conv1_out = tf.nn.leaky_relu(batch_norm1,
                                        name="conv1_out")
            
            # 64x64x64-> 32x32x128 
            conv2 = tf.compat.v1.layers.conv2d(inputs=conv1_out,
                                    filters=128,
                                    kernel_size=[5, 5],
                                    strides=[2, 2],
                                    padding="SAME",
                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                    name='conv2')
            batch_norm2 = tf.compat.v1.layers.batch_normalization(conv2,
                                                        training=True,
                                                        epsilon=self.EPSILON,
                                                        name='batch_norm2')
            conv2_out = tf.nn.leaky_relu(batch_norm2,
                                        name="conv2_out")
            
            # 32x32x128 -> 16x16x256  
            conv3 = tf.compat.v1.layers.conv2d(inputs=conv2_out,
                                    filters=256,
                                    kernel_size=[5, 5],
                                    strides=[2, 2],
                                    padding="SAME",
                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                    name='conv3')
            batch_norm3 = tf.compat.v1.layers.batch_normalization(conv3,
                                                        training=True,
                                                        epsilon=self.EPSILON,
                                                        name='batch_norm3')
            conv3_out = tf.nn.leaky_relu(batch_norm3,
                                        name="conv3_out")
            
            # 16x16x256 -> 16x16x512
            conv4 = tf.compat.v1.layers.conv2d(inputs=conv3_out,
                                    filters=512,
                                    kernel_size=[5, 5],
                                    strides=[1, 1],
                                    padding="SAME",
                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                    name='conv4')
            batch_norm4 = tf.compat.v1.layers.batch_normalization(conv4,
                                                        training=True,
                                                        epsilon=self.EPSILON,
                                                        name='batch_norm4')
            conv4_out = tf.nn.leaky_relu(batch_norm4,
                                        name="conv4_out")
            
            # 16x16x512 -> 8x8x1024
            conv5 = tf.compat.v1.layers.conv2d(inputs=conv4_out,
                                    filters=1024,
                                    kernel_size=[5, 5],
                                    strides=[2, 2],
                                    padding="SAME",
                                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=self.WEIGHT_INIT_STDDEV),
                                    name='conv5')
            batch_norm5 = tf.compat.v1.layers.batch_normalization(conv5,
                                                        training=True,
                                                        epsilon=self.EPSILON,
                                                        name='batch_norm5')
            conv5_out = tf.nn.leaky_relu(batch_norm5,
                                        name="conv5_out")

            flatten = tf.reshape(conv5_out, (-1, 8*8*1024))
            logits = tf.compat.v1.layers.dense(inputs=flatten,
                                    units=1,
                                    activation=None)
            out = tf.sigmoid(logits)
            return out, logits


    def model_loss(self, input_real, input_z, output_channel_dim):
        g_model = self.generator(input_z, output_channel_dim, True)

        noisy_input_real = input_real + tf.compat.v1.random_normal(shape=tf.shape(input_real),
                                                        mean=0.0,
                                                        stddev=random.uniform(0.0, 0.1),
                                                        dtype=tf.float32)
        
        d_model_real, d_logits_real = self.discriminator(noisy_input_real, reuse=False)
        d_model_fake, d_logits_fake = self.discriminator(g_model,          reuse=True)
        
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * random.uniform(0.9, 1.0)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake) ))
        d_loss = tf.reduce_mean(0.5 * (d_loss_real + d_loss_fake))
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
        return d_loss, g_loss


    def model_optimizers(self, d_loss, g_loss):
        t_vars = tf.compat.v1.trainable_variables()
        g_vars = [var for var in t_vars if var.name.startswith("generator")]
        d_vars = [var for var in t_vars if var.name.startswith("discriminator")]
        
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        gen_updates = [op for op in update_ops if op.name.startswith('generator') or op.name.startswith('discriminator')]
        
        with tf.control_dependencies(gen_updates):
            d_train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.LR_D, beta1=self.BETA1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.LR_G, beta1=self.BETA1).minimize(g_loss, var_list=g_vars)  
        return d_train_opt, g_train_opt

    def model_inputs(self, real_dim, z_dim):
        inputs_real = tf.compat.v1.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
        inputs_z = tf.compat.v1.placeholder(tf.float32, (None, z_dim), name="input_z")
        learning_rate_G = tf.compat.v1.placeholder(tf.float32, name="lr_g")
        learning_rate_D = tf.compat.v1.placeholder(tf.float32, name="lr_d")
        return inputs_real, inputs_z, learning_rate_G, learning_rate_D


    def show_samples(self, sample_images, name, epoch):
        figure, axes = plt.subplots(1, len(sample_images), figsize = (self.IMAGE_SIZE, self.IMAGE_SIZE))
        for index, axis in enumerate(axes):
            axis.axis('off')
            image_array = sample_images[index]
            axis.imshow(image_array)
            image = Image.fromarray(image_array)
            image.save(name+"_"+str(epoch)+"_"+str(index)+".png") 
        plt.savefig(name+"_"+str(epoch)+".png", bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.close()

    def test(self, sess, input_z, out_channel_dim, epoch):
        example_z = np.random.uniform(-1, 1, size=[self.SAMPLES_TO_SHOW, input_z.get_shape().as_list()[-1]])
        samples = sess.run(self.generator(input_z, out_channel_dim, False), feed_dict={input_z: example_z})
        sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]
        self.show_samples(sample_images, self.OUTPUT_DIR + "samples", epoch)



    def summarize_epoch(self, epoch, sess, d_losses, g_losses, input_z, data_shape, saver):
        print("\nEpoch {}/{}".format(epoch, self.EPOCHS),
            "\nD Loss: {:.5f}".format(np.mean(d_losses[-self.MINIBATCH_SIZE:])),
            "\nG Loss: {:.5f}".format(np.mean(g_losses[-self.MINIBATCH_SIZE:])))
        fig, ax = plt.subplots()
        plt.plot(d_losses, label='Discriminator', alpha=0.6)
        plt.plot(g_losses, label='Generator', alpha=0.6)
        plt.title("Losses")
        plt.legend()
        plt.savefig(self.OUTPUT_DIR + "losses_" + str(epoch) + ".png")
        # plt.show()
        plt.close()
        saver.save(sess, self.OUTPUT_DIR + "model_" + str(epoch) + ".ckpt")
        self.test(sess, input_z, data_shape[3], epoch)

    def get_batch(self, dataset):
        files = random.sample(dataset, self.BATCH_SIZE)
        batch = []
        for file in files:
            if random.choice([True, False]):
                batch.append(np.asarray(Image.open(file).transpose(Image.FLIP_LEFT_RIGHT)))
            else:
                batch.append(np.asarray(Image.open(file)))                     
        batch = np.asarray(batch)
        normalized_batch = (batch / 127.5) - 1.0
        return normalized_batch, files

    def train(self, data_shape, epoch, checkpoint_path):
        input_images, input_z, lr_G, lr_D = self.model_inputs(data_shape[1:], self.NOISE_SIZE)
        d_loss, g_loss = self.model_loss(input_images, input_z, data_shape[3])
        d_opt, g_opt = self.model_optimizers(d_loss, g_loss)
        
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            if checkpoint_path is not None:
                saver.restore(sess, checkpoint_path)
                
            iteration = 0
            d_losses = []
            g_losses = []
            
            for epoch in range(self.EPOCH, self.EPOCHS):        
                epoch += 1
                epoch_dataset = self.DATASET.copy()
                
                for i in range(self.MINIBATCH_SIZE):
                    iteration_start_time = time.time()
                    iteration += 1
                    batch_images, used_files = self.get_batch(epoch_dataset)
                    [epoch_dataset.remove(file) for file in used_files]
                    
                    batch_z = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, self.NOISE_SIZE))
                    _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: self.LR_D})
                    _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: self.LR_G})
                    d_losses.append(d_loss.eval({input_z: batch_z, input_images: batch_images}))
                    g_losses.append(g_loss.eval({input_z: batch_z}))
                    
                    elapsed_time = round(time.time()-iteration_start_time, 3)
                    remaining_files = len(epoch_dataset)
                    print("\rEpoch: " + str(epoch) +
                        ", iteration: " + str(iteration) + 
                        ", d_loss: " + str(round(d_losses[-1], 3)) +
                        ", g_loss: " + str(round(g_losses[-1], 3)) +
                        ", duration: " + str(elapsed_time) + 
                        ", minutes remaining: " + str(round(remaining_files/self.BATCH_SIZE*elapsed_time/60, 1)) +
                        ", remaining files in batch: " + str(remaining_files)
                        , sep=' ', end=' ', flush=True)
                    
                self.summarize_epoch(epoch, sess, d_losses, g_losses, input_z, data_shape, saver)

    def run_train(self):

        # Training
        with tf.Graph().as_default():
            self.train(data_shape=(self.DATASET_SIZE, self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
                epoch=self.EPOCH,
                checkpoint_path=None)


    def generate(self):

        # Training
        #import keras
        path_file_checkpoint_meta = 'C:\\project\\breaker\\breaker_generator\\script\\model_1.ckpt.meta'
        path_file_checkpoint_weight = 'C:\\project\\breaker\\breaker_generator\\script\\model_1.ckpt.data-00000-of-00001' 
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, path_file_checkpoint_weight)
            # new_saver = tf.compat.v1.train.import_meta_graph(path_file_checkpoint_meta)
            # new_saver.restore(sess, path_file_checkpoint_weight)
            #new_saver.save('my_model.h5')
        #model = keras.models.load_model('C:\\project\\breaker\\breaker_generator\\script\\model_1.ckpt.data-00000-of-00001')