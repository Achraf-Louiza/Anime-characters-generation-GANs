from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Dropout,
    Flatten, Conv2DTranspose, MaxPooling2D, 
    UpSampling2D, Reshape, BatchNormalization, LeakyReLU
)
from tensorflow.keras.models import Model

class Discriminator:
    '''
    A deconvolution discriminator for DCGAN
    '''
    def __init__(self, img_shape):
        
        inputs = Input(shape=img_shape)
        
        x = Conv2D(64, kernel_size=5, strides=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(64, kernel_size=5, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        # add dilated convolutions
        x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
       
        x = Flatten()(x)
        output = Dense(1)(x)
        
        self.model = Model(inputs, output, name='deconv-discriminator')