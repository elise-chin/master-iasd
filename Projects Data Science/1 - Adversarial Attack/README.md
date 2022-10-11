# Project 1-2: Adversarial Robustness

**Group**: Clément Chauvet, Elise Chin, Mathilde Da Cruz

**Objective**: Train the most robust model to adversarial attacks using GAN.

Idea based on the paper [Rob-GAN: Generator, Discriminator, and Adversarial Attacker](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Rob-GAN_Generator_Discriminator_and_Adversarial_Attacker_CVPR_2019_paper.pdf) by Liu et al. (2018)

## Motivation

Since adversarial training is already known as a very good defense method, we wanted to use it in some way. But our reflection led us to think that for a good defense, we needed a good attack and vice versa. That's how we thought about some system in which the defense part and the attack part could train each other. Also, we thought that a good defense could classify any image, even one that is different from its training set. In other words, images from the largest possible part from the space of images. For that, we need a generator.
We rapidly found a paper that was very close from our idea, and then we tried to explore it, and implement it, based on the github code from the authors.

## Experiments

The main idea is to combine adversarial training and training on fake images.

### GANs for adversarial robustness

GANs are often used for image generation. Here, the generator will be jointly trained with the discriminator, and will try to fool the discriminator. Different models for the generator can be found in the `gen_models` directory.

The discriminator of the GAN is the network we want to make robust to adversarial attacks. It is the defense part. It tries to correctly classify attacked images and fake images, and helps to improve the generator by also classifying real from fake images. Models for the discriminator can be found in the `dis_models` directory.

### Training

One interesting thing to note with adversarial training is that there is a gap (bigger than usual) of performance between the training and the test set. This could be explained by the fact the network only visits a small region of the space of images. This is what intuitively gives the interest to the generator.

We trained the GAN on two benchmarks datasets: CIFAR-10 and MNIST. RobGAN was trained for 32 epochs on CIFAR-10 and 50 epochs on MNIST. The pipeline can be found in the `train_MNIST.ipynb` notebook, and shows the execution on MNIST only. 

The original RobGAN was trained on CIFAR-10. The main difficulties encountered was to understand the code, updating it with a more recent PyTorch's version and adapting it for MNIST.

### Results

#### CIFAR-10

Below are images generated after the 1st and 30th epoch on CIFAR-10. The network first learns to generate a white spot at a precise location for all samples images, then only generates gray images.

![Images generated after 1st epoch on CIFAR](./CIFAR_out/sample_epoch_0.png)

![Images generated after 30 epochs on CIFAR](./CIFAR_out/sample_epoch_30.png)

#### MNIST

We observe the same behaviour on MNIST images too.

![Images generated after 1st epoch on MNIST](./MNIST_out/sample_epoch_0.png)

![Images generated after 30 epochs on MNIST](./MNIST_out/sample_epoch_30.png)

We encountered a convergence problem on the loss of the generator. After a few batch it becomes clear that the weight of the generator are not converging anymore and we fall into a mode collapse. After a few epochs all images are uniformly gray and no more progress is made.