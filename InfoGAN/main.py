# Code from https://github.com/pianomania/infoGAN-pytorch/blob/master/main.py
from model import *
from trainer import Traniner

fe = FrontEnd()
d = D()
q = Q()
g = G()

for i in  [fe, d, q, g]:
    i.cuda()
    i.apply(weights_init)

    trainer = Trainer(g, fe, d, q)
    trainer.train()