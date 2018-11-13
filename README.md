## Hessian Flow: A Library for Hessian Based Algorithms in Machine Learning

HessianFlow is a library that supports various second-order based methods
developed for training neural network models that works in conjunction with pytorch.
The library currently supports utility functions to compute Hessian spectrum of different neural network
models.

## Model Training

python train.py --name cifar10 --epochs 90 --lr 0.02 --lr-decay 0.2 --lr-decay-epoch 30 60 --arch c1

## Hessian Spectrum Computation

python hessian_eig_driver.py --name cifar10 --arch c1 --resume model_param.pkl --eigen-type full

## Track Eigen During the whole training procedure of ResNet on cifar10

python train_resnet.pt --name cifar10 --epoch 160 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60 90 120 

## Citation
HessianFlow has been developed as part of the following papers. We appreciate it if you could please
cite these if you found the library useful for your work:


* Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. Mahoney. Hessian-based Analysis of Large Batch Training and Robustness to Adversaries, NIPS'18 (arXiv:1802.08241)
* Z. Yao, A. Gholami, K. Keutzer, M. Mahoney. Large Batch Size Training of Neural Networks with Adversarial Training and Second-Order Information, arxiv:1810.01021 
