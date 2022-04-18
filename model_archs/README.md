# GAN Architectures

Each file (besides this one) details the architecture behind the discriminator, generator, and full GAN (and q model in the case of the InfoGAN) for each type of GAN in the study. Here are is the "table of contents":

- [ACGAN](acgan_arch.md)
- [CGAN](cgan_arch.md)
- [GAN](gan_arch.md)
- [InfoGAN](infogan_arch.md)
- [SGAN](sgan_arch.md)


## Explanation of Model Summary

A sample model summary of the CGAN discriminator is shown below to describe what's really going on.


```
Discriminator:
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 1)]          0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1, 50)        200         input_1[0][0]                    
__________________________________________________________________________________________________
dense (Dense)                   (None, 1, 744)       37944       embedding[0][0]                  
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 744, 1, 1)]  0                                            
__________________________________________________________________________________________________
reshape (Reshape)               (None, 744, 1, 1)    0           dense[0][0]                      
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, 744, 1, 2)    0           input_2[0][0]                    
                                                                 reshape[0][0]                    
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 372, 1, 128)  2432        concatenate[0][0]                
__________________________________________________________________________________________________
leaky_re_lu (LeakyReLU)         (None, 372, 1, 128)  0           conv2d[0][0]                     
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 186, 1, 128)  147584      leaky_re_lu[0][0]                
__________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)       (None, 186, 1, 128)  0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 93, 1, 128)   147584      leaky_re_lu_1[0][0]              
__________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)       (None, 93, 1, 128)   0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
flatten (Flatten)               (None, 11904)        0           leaky_re_lu_2[0][0]              
__________________________________________________________________________________________________
dropout (Dropout)               (None, 11904)        0           flatten[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 1)            11905       dropout[0][0]                    
==================================================================================================
Total params: 347,649
Trainable params: 0
Non-trainable params: 347,649
__________________________________________________________________________________________________
```

The first column lists the layer type and name. The second column lists the dimensions of the output. The third column lists the amount of parameters. The fourth column lists which column the layer is connected to.

### Definitions

- `InputLayer` (first column): this is a layer that takes in the input (training data)
- `Embedding` (first column): this layer embeds the input into a continuous and latent space of the specified dimension.
- `Dense` (first column): the Dense layer performs a matrix multiple of learned weights with the provided input and adds a bias. Then an activation function is applied on the output (nonlinear). In machine learning, this is a fully-connected NN layer.
- `Reshape` (first column): Changes the dimensions of the output
- `Concatenate` (first column): Combines the results of the previous two layers.
- `Conv2D` (first column): A convolutional layer: see https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks for more information. This is similar to a `Dense` layer, except that learned filters are applied over a image-like input.
- `LeakyReLU` (first column): This is the function `x if x > 0; 0.1x if x <= 0`. This is a non linear activation layer.
- `Flatten` (first column): This layer flattens the input to be one-dimensional.
- `Dropout` (first column): This layer zeroes out some of the weights during training.


For the number of convolution layers, kernel size, etc:

For brevity reasons, these values can be found in the code (e.g., https://github.com/peasant98/GAN-energy-modeling/blob/main/gan/gan.py#L62)
