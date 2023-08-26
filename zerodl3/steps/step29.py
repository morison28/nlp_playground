#
# 勾配降下法とニュートン法の比較
#

# 勾配降下法では計算回数が多く必要になる
# ニュートン法では7回計算すれば最適化にたどり着く

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero.core_simple import Variable


def f(x):
    return x**4 - 2*x**2

def gx2(x):
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()

    y.backward()

    x.data -= x.grad / gx2(x.data)
