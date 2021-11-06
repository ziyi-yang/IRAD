# Anomaly Detection with Domain Adaptation

The implementations for the paper
[Anomaly Detection with Domain Adaptation](https://arxiv.org/pdf/2006.03689.pdf).

To obtain IRAD performance on the MNIST-->USPS experiment (e.g. class 8), run:
```zsh
python main.py --model IRAD --src mnist --tgt usps --l_dim 64 --nb_epochs 15 --tgt_num 50 --label 8 --lr 1e-4
```
