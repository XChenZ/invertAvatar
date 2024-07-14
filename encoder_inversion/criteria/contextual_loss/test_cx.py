import torch
import os
import sys
sys.path.append(os.path.abspath('.'))
from criterion.contextual_loss.modules import ContextualBilateralLoss, ContextualLoss
from criteria.cx_loss import CXLoss
import time

contextual_loss = ContextualBilateralLoss(use_vgg=True).to('cuda').eval()
cx_loss = ContextualLoss(use_vgg=True).to('cuda').eval()


diy_loss = CXLoss().to('cuda').eval()

a = torch.randn(2, 3, 224, 224).cuda()
b = torch.randn(2, 3, 224, 224).cuda()

t1 = time.time()
c = diy_loss(a, b)
t2 = time.time()
print(t2 - t1)

a = torch.randn(2, 3, 224, 224).cuda()
b = torch.randn(2, 3, 224, 224).cuda()

t1 = time.time()
c = diy_loss(a, b)
t2 = time.time()
print(t2 - t1)

a = torch.randn(2, 3, 224, 224).cuda()
b = torch.randn(2, 3, 224, 224).cuda()

t1 = time.time()
c = contextual_loss(a, b)
t2 = time.time()
print(t2 - t1)

a = torch.randn(2, 3, 224, 224).cuda()
b = torch.randn(2, 3, 224, 224).cuda()

t1 = time.time()
c = contextual_loss(a, b)
t2 = time.time()
print(t2 - t1)

a = torch.randn(2, 3, 224, 224).cuda()
b = torch.randn(2, 3, 224, 224).cuda()

t1 = time.time()
c = cx_loss(a, b)
t2 = time.time()
print(t2 - t1)

a = torch.randn(2, 3, 224, 224).cuda()
b = torch.randn(2, 3, 224, 224).cuda()

t1 = time.time()
c = cx_loss(a, b)
t2 = time.time()
print(t2 - t1)


