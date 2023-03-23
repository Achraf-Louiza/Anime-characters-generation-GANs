# Automatic Anime Characters Creation with GANs
This is a project that aims to automatically create anime characters using Generative Adversarial Networks (GANs). The project is based on the paper "Towards the Automatic Anime Characters Creation with Generative Adversarial Networks" by Takuhiro Kaneko, which can be found [here](https://arxiv.org/pdf/1708.05509.pdf).

## Generative Adversarial Networks (GANs)
GANs are a type of neural network that consists of two parts: a generator and a discriminator. The generator creates new data that is similar to a training set, while the discriminator tries to distinguish between the generated data and the real data. The two parts are trained simultaneously, with the generator trying to fool the discriminator and the discriminator trying to correctly identify the real data.

## Anime Character Dataset
The dataset used in this project is the Anime Face Dataset, which consists of over 63,000 anime face images. The dataset can be found [here](https://www.kaggle.com/datasets/splcher/animefacedataset).

## Project Structure
The project is structured as follows:
- **data/**: Contains the dataset. Data is available on kaggle.
- **models/**: Contains the GAN models implemented in the project.  
         - **dcgan/**: Deep convolution GAN implementation  
         - **wgangp/**: Wasserstein GAN with gradient penalty  
         - **sagan/**: Self attention GAN  
- **functions/**: Contains utility functions such as data reading
- **anime-character-generation.ipynb**: Notebook containing code to train gan on anime characters dataset

## References
[[1]](https://arxiv.org/pdf/1708.05509.pdf) Kaneko, T. (2017). Towards the Automatic Anime Characters Creation with Generative Adversarial Networks.   
[[2]](https://arxiv.org/abs/1406.2661) Goodfellow, Ian, et al. "Generative Adversarial Networks." arXiv preprint arXiv:1406.2661 (2014)   
[[3]](https://arxiv.org/abs/1511.06434) Radford, Alec, et al. "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." arXiv preprint arXiv:1511.06434 (2015)   
[[4]](https://arxiv.org/abs/1701.07875) Arjovsky, Martin, et al. "Wasserstein Generative Adversarial Networks." arXiv preprint arXiv:1701.07875 (2017)   
[[5]](https://www.kaggle.com/splcher/animefacedataset) Anime Face Dataset.    
