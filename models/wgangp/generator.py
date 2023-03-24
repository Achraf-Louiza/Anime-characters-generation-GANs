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
    
    def __init__(self, latent_dim, img_shape):
        
        n_debut = img_shape[0]//2**3
        inputs = Input(shape=latent_dim)
        
        x = Dense(n_debut*n_debut*256)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape((n_debut, n_debut, 256))(x)
        
        x = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

      
        output = Conv2DTranspose(3, kernel_size=5, strides=1, padding='same', activation='tanh')(x)
        
        self.model = Model(inputs, output, name='conv-generator')
    