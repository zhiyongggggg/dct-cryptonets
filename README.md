# Compressed Homomorphic Encryption Neural Network Project

In order to improve the latency of deep homomorphic encryption networks (which are extremely slow) we attempt to change
the input representation so that we feed compressed frequency information (DCT or PCA coefficients) into the 
encrypted neural network rather than the raw images. This hopefully reduces the size of the encrypted ciphertext which
would reduce latency during processing. We will play with the percentages of low/high frequency components fed into
the network. There is also a potential for processing only low frequency information through the encrypted network while
processing the high frequency information through a standard un-encrypted network. This uses the understanding that
low frequency components of an image are more important for machine understanding (and even general human understanding)
compared to high frequency components.


## Code Sources
The following repositories were used as a backbone
1. https://github.com/xiangyu8/PT-MAP-sf
2. https://github.com/kaix90/DCTNet


## Installations
### Other than pip installs
Need libjpeg-turbo:
```angular2html
conda install libjpeg-turbo --channel=conda-forge
```
Forcing conda to install a cuda compatible pytorch:
```angular2html
conda install pytorch=*=*cuda* torchvision cudatoolkit -c pytorch
```


## Download MNIST Dataset
https://www.kaggle.com/datasets/scolianni/mnistasjpg
saved to /home/arjunroy/datasets/mnist