## Hessian Flow: A Library for Hessian Based Algorithms in Machine Learning

HessianFlow is a pytorch library for Hessian based analysis of neural network models that could be used in conjunction with pytorch.
The library currently supports utility functions to compute Hessian spectrum of different neural network
models.


## Example Usage
After installing pytoch, the following command trains a ResNet on Cifar-10 and prints the dominant Hessian eigenvalue at every epoch:

python train_resnet.pt --name cifar10 --epoch 160 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60 90 120 

You can also use the library to compute Hessian spectrum at different snapshots after training is finished. For instance, here we first
train a simple network:

python train.py --name cifar10 --epochs 90 --lr 0.02 --lr-decay 0.2 --lr-decay-epoch 30 60 --arch c1

And afterwards we load the checkpoint and compute the Hessian spectrum at that point:

python hessian_eig_driver.py --name cifar10 --arch c1 --resume model_param.pkl --eigen-type full


## Citation
HessianFlow has been developed as part of the following papers. We appreciate it if you would please
cite the following if you found the library useful for your work:


* Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. Mahoney. Hessian-based Analysis of Large Batch Training and Robustness to Adversaries, NIPS'18 [PDF](https://arxiv.org/pdf/1802.08241)
* Z. Yao, A. Gholami, K. Keutzer, M. Mahoney. Large Batch Size Training of Neural Networks with Adversarial Training and Second-Order Information, [PDF](https://arxiv.org/pdf/1810.01021.pdf)


## Extensions
- Please see Noah's [repo](https://github.com/noahgolmant/pytorch-hessian-eigenthings) on how to compute the other eigenvalues of the Hessian