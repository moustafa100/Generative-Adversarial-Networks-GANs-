#!/usr/bin/env python
# coding: utf-8

# In[11]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
from keras.layers import Input,Dense,Conv2D,Conv2DTranspose,Flatten,Reshape
from keras.datasets import mnist
from keras.models import Model
import numpy as np 
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from tensorflow.keras import backend as k 
from PIL import Image
from keras.layers import BatchNormalization,Activation,LeakyReLU
from tensorflow.keras.models import load_model
import os
import argparse


# In[2]:


def build_generator(inputs,img_size):
    img_resize=img_size//4
    # network parameters
    kernel=5
    layer_filter=[128,64,32,1]
    
    x=Dense(img_resize*img_resize*layer_filter[0])(inputs)
    x=Reshape((img_resize,img_resize,layer_filter[0]))(x)
    
    for filters in layer_filter:
        # first two convolution layers use strides = 2
        # the last two use strides = 1
        if filters > layer_filter[-2]:
            strides=2
        else:
            strides=1
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2DTranspose(filters=filters,kernel_size=kernel,
                          strides=strides,padding='same')(x)
    x=Activation('sigmoid')(x)
    generator=Model(inputs,x,name='generator')
    return generator
    


# In[3]:


def build_discriminator(inputs):
    kernel=5
    layer_filters=[32,64,128,256]
    x=inputs
    for filters in layer_filters:
        # first 3 convolution layers use strides = 2
        # last one uses strides = 1
        if filters == layer_filters[-1]:
            strides=1
        else :
            strides=2
        x=LeakyReLU(alpha=.2)(x)
        x=Conv2D(kernel_size=kernel,strides=strides,
                 filters=filters,padding='same')(x)
    x=Flatten()(x)
    x=Dense(1)(x)
    x=Activation('sigmoid')(x)
    discri=Model(inputs,x,name='discriminator')
    return discri


# In[4]:


def train(models,x_train,params):
    # the GAN component models
    generator,discri,adver=models
    # network parameters
    batch,latent,train_steps,model_name=params
    # the generator image is saved every 500 steps
    save_interval=500
    # noise vector to see how the generator output evolves during training
    noise_input=np.random.uniform(1.0,-1.0,size=[16,latent])
    # number of elements in train dataset
    train_size=x_train.shape[0]
    for i in range(train_steps):
        # randomly pick real images from dataset
        rand_indecis=np.random.randint(0,train_size,size=batch)
        real_img=x_train[rand_indecis]
        # generate fake images from noise using generator
        noise=np.random.uniform(1.0,-1.0,size=[batch,latent])
        # generate fake images
        fake_img=generator.predict(noise)
        # real + fake images = 1 batch of train data
        x=np.concatenate((real_img,fake_img))
        # label real and fake images
        y=np.ones([2*batch,1])
        y[batch:,:]=0.0
        # train discriminator network, log the loss and accuracy
        loss,acc=discri.train_on_batch(x,y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)
        # generate noise using uniform distribution
        noise=np.random.uniform(1.0,-1.0,size=[batch,latent])
        #label fake images as real or 1.0
        y=np.ones([batch,1])
        loss,acc=adver.train_on_batch(noise,y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            # plot generator images on a periodic basis
            plot_images(generator,
                    noise_input=noise_input,
                    show=False,
                    step=(i + 1),
                    model_name=model_name)
    # future MNIST digit generation
    generator.save(model_name + ".h5")
        


# In[5]:


def plot_images(generator,
                noise_input,
                show=False,
                step=0,
                model_name="gan"):
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2, 2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')


# In[6]:


def build_and_train_model():
    (x_train,_),(_,_)=mnist.load_data()
    img_size=x_train.shape[1]
    x_train=np.reshape(x_train,[-1,img_size,img_size,1])
    x_train=x_train.astype('float32')/255
    model_name='dcgan_mnist'
    # network parameters
    latent=100
    batch=64
    train_steps=40000
    lr=2e-4
    decay=6e-8
    input_shape=(img_size,img_size,1)
    # build discriminator model
    inputs=Input(shape=input_shape,name='discriminator_input')
    discri=build_discriminator(inputs)
    optimizer=RMSprop(lr=lr,decay=decay)
    discri.compile(loss='binary_crossentropy',optimizer=optimizer,
                            metrics=['accuracy'])
    discri.summary()
    
    
    # build generator model
    
    input_shape=(latent,)
    inputs=Input(shape=input_shape,name='z_input')
    generator=build_generator(inputs,img_size)
    optimizer=RMSprop(lr=lr*.5,decay=decay*.5)
    discri.trainable=False
    # adversarial = generator + discriminator
    adver=Model(inputs,discri(generator(inputs)),
                name=model_name)
    adver.compile(loss='binary_crossentropy',optimizer=optimizer,
                            metrics=['accuracy'])
    adver.summary()
    
    models = (generator, discri, adver)
    params = (batch, latent, train_steps, model_name)
    train(models, x_train, params)
    


# In[7]:


def test_generator(generator):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
    plot_images(generator,
                noise_input=noise_input,
                show=True,
                model_name="test_outputs")


# In[8]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    else:
        build_and_train_model()


# In[ ]:




