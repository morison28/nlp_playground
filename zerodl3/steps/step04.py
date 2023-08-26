import numpy as np

from step01 import Variable
from step02_03 import Function, Square, Exp


def numerical_diff(f: Function, x: Variable, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

if __name__ == '__main__':
    # 微分試してみる
    f = Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)
    # 合成関数の微分も簡単にできる

    def f(x: Variable):
        A = Square()
        B = Exp()
        C = Square()
        return C(B(A(x)))

    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x)
    print(dy)