if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero.core_simple import Variable
# from dezero.utils import plot_dot_graph

#
# ローゼンブロック関数を最適化する
#

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001 # 学習率
iters = 10000

x0s = [x0.data]
x1s = [x1.data]

for i in range(iters):
    # if :
    #     print(x0, x1)
    # print(x0, x1)
    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

    _x0 = x0.data
    _x1 = x1.data

    x0s.append(_x0.copy())
    x1s.append(_x1.copy())


import matplotlib.pyplot as plt
import japanize_matplotlib

fig = plt.figure(figsize=(5,5))
ax = fig.subplots()
ax.scatter(x0s, x1s)
ax.grid(color='black', alpha=0.5)
ax.set_xlim(-2,)
fig.suptitle('10000回イテレーションして最適化するといい感じになる')
fig.savefig('step28_rosenbrock_optim.png')
y = rosenbrock(x0, x1)
y.backward()
# print(x0.grad, x1.grad)