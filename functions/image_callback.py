import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define a function to generate a batch of images
def generate_images(generator, latent_dim, n_samples):
    # Generate random input for the generator
    input_latent = tf.random.normal([n_samples, latent_dim])
    # Generate images from the input
    generated_images = generator(input_latent, training=False)

    return generated_images.numpy()

# Define a custom callback to generate and save some sample images every few epochs
class ImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, latent_dim, save_dir):
        super().__init__()
        self.generator = generator
        self.latent_dim = latent_dim
        self.save_dir = save_dir
    
    def on_epoch_end(self, epoch, logs=None):
        # Generate and save some sample images
        generated_images = generate_images(self.generator, self.latent_dim, 30)
        fig, axs = plt.subplots(3, 10, figsize=(13,4))
        for i, ax in enumerate(axs.flat):
            ax.imshow(generated_images[i])
            ax.axis('off')
        plt.savefig(f'{self.save_dir}/generated_epoch{epoch}.png')
        plt.show()
        plt.close(fig)

