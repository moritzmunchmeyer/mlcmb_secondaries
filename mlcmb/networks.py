import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K 

import config



class EstimatorNet():
    
    def __init__(self,params):
        self.params = params
       

    ###################### start with a simple Unet from google tutorial
    #https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb

    def unet_simple(self,img_shape,channels_out):

        filterdivisor = 4

        inputs = layers.Input(shape=img_shape)
        # 256

        encoder0_pool, encoder0 = self.encoder_block_unet(inputs, int(32/filterdivisor))
        # 128

        encoder1_pool, encoder1 = self.encoder_block_unet(encoder0_pool, int(64/filterdivisor))
        # 64

        encoder2_pool, encoder2 = self.encoder_block_unet(encoder1_pool, int(128/filterdivisor))
        # 32

        encoder3_pool, encoder3 = self.encoder_block_unet(encoder2_pool, int(256/filterdivisor))
        # 16

        encoder4_pool, encoder4 = self.encoder_block_unet(encoder3_pool, int(512/filterdivisor))
        # 8

        center = self.conv_block_unet(encoder4_pool, int(1024/filterdivisor))
        # center

        decoder4 = self.decoder_block_unet(center, encoder4, int(512/filterdivisor))
        # 16

        decoder3 = self.decoder_block_unet(decoder4, encoder3, int(256/filterdivisor))
        # 32

        decoder2 = self.decoder_block_unet(decoder3, encoder2, int(128/filterdivisor))
        # 64

        decoder1 = self.decoder_block_unet(decoder2, encoder1, int(64/filterdivisor))
        # 128

        decoder0 = self.decoder_block_unet(decoder1, encoder0, int(32/filterdivisor))
        # 256

        #outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
        outputs = layers.Conv2D(channels_out, (1, 1), activation='linear')(decoder0)

        return inputs,outputs



    def conv_block_unet(self,input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        #encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        #encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block_unet(self,input_tensor, num_filters):
        encoder = self.conv_block_unet(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        return encoder_pool, encoder

    def decoder_block_unet(self,input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        #decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        #decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        #decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder
    
    
    
    
    
    
    
