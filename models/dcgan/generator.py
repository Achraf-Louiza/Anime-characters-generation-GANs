from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Dropout,
    Conv2DTranspose, MaxPooling2D, UpSampling2D, 
    Reshape, LeakyReLU, BatchNormalization
)
from tensorflow.keras.models import Model

class Generator:
    '''
    A convolution generator for DCGAN
    '''
    
    def __init__(self, latent_dim):
        
        inputs = Input(shape=latent_dim)
        
        x = Dense(5*5*256)(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((5, 5, 256))(x)
        
        x = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        output = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(x)
        
        self.model = Model(inputs, output, name='conv-generator')
    