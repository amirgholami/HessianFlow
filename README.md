## Hessian Flow: A Library for Hessian Based Algorithms in Machine Learning

HessianFlow is a library that supports various second-order based methods
developed for training neural network models that works in conjunction with pytorch.
The library currently supports utility functions to compute Hessian spectrum of different neural network
models.

## ABSA: Adaptive Batch Size with Adversarial training:
This method uses second order information to adaptively increase batch size when SGD training gets to flatter
landscapes. Details can be found in [this paper](https://arxiv.org/pdf/1810.01021.pdf). Example codes to run
with ABSA:

python train.py --name cifar10 --epochs 90 --lr 0.02 --lr-decay 0.2 --lr-decay-epoch 30 60 --arch c1

python train.py --arch ResNet --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 30 60 80 --large-ratio 128 --method absa



## Hessian Spectrum Computation
This code computes the Hessian spectrum for a saved neural network model.

python hessian_eig_driver.py --name cifar10 --arch c1 --resume model_param.pkl --eigen-type full

## Track Eigen During the whole training procedure of ResNet on cifar10
One could simply modify the above example to track the spectrum of the Hessian throughout training:

python train_resnet.pt --name cifar10 --epoch 160 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60 90 120 

## Citation
If you found the library useful, we apprecaite it if you cite the following work:

* Z. Yao, A. Gholami, K. Keutzer, M. Mahoney. Large Batch Size Training of Neural Networks with Adversarial Training and Second-Order Information, arxiv:1810.01021 
* Z. Yao, A. Gholami, Q. Lei, K. Keutzer, M. Mahoney. Hessian-based Analysis of Large Batch Training and Robustness to Adversaries, NeurIPS'18 (arXiv:1802.08241)
