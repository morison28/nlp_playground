import weakref

import numpy as np
from memory_profiler import profile


class Variable:
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.name = name
        self.grad = None # 逆伝搬で受け取った勾配を保持
        self.creator = None # 自身を生み出した関数を記憶
        self.generation = 0

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return f"variable({p})"

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        # 勾配の初期化を省略するためのコード
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        # 世代順に関数を並べる
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # リストの末尾の取得
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(f.inputs, gxs):
                # step14: 同じオブジェクトを足したときに勾配が上書きされることを防ぐ
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)
            # step18
            # ほとんどの場合しりたいのは1層目の入力の勾配であることが多い
            # したがって関数の出力の勾配はリセットしてメモリから解放する
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

        # 再帰を使うstep7 
        # f = self.creator # 自身を生み出した関数
        # if f is not None:
        #     x = f.input # その関数の入力
        #     x.grad = f.backward(self.grad) # 勾配はその関数の微分に自身の勾配をかけたもの
        #     x.backward() # これを再帰的に呼ぶことで自動化する
    
    def cleargrad(self):
        # step14: 微分のリセット
        self.grad = None

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        # 循環参照の解消(step17)
        # 参照カウントを増やすことなく別のオブジェクトを参照
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def add(x0, x1):
    return Add()(x0, x1)

def square(x: Variable):
    return Square()(x)

def exp(x: Variable):
    return Exp()(x)

@profile
def calc_memory():
    for _ in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))


if __name__ == '__main__':
    
    # z = x^2 + y^2 の微分
    # x = Variable(np.array(2))
    # a = square(x)
    # y = add(square(a), square(a))
    # y.backward()
    # print(y.data)
    # print(x.grad) # OK!

    v = Variable(np.array([[1,2,3], [4,5,6]]))
    print(v)

    # calc_memory()
""" 弱参照でメモリ減らしたVer
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
   126     84.8 MiB     84.8 MiB           1   @profile
   127                                         def calc_memory():
   128     85.4 MiB      0.0 MiB          11       for _ in range(10):
   129     85.4 MiB      0.0 MiB          10           x = Variable(np.random.randn(10000))
   130     85.4 MiB      0.6 MiB          10           y = square(square(square(x)))
"""

""" 減らさないとき(step16)
ネットワークが小さいからしらんが効果は小さそう
でも上より減っている
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
   122     82.9 MiB     82.9 MiB           1   @profile
   123                                         def calc_memory():
   124     86.1 MiB      0.0 MiB          11       for _ in range(10):
   125     85.9 MiB      0.8 MiB          10           x = Variable(np.random.randn(10000))
   126     86.1 MiB      2.4 MiB          10           y = square(square(square(x)))
"""

""" step18: retain_grad = Falseで勾配の保持をやめる
Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
   132     80.8 MiB     80.8 MiB           1   @profile
   133                                         def calc_memory():
   134     81.5 MiB      0.0 MiB          11       for _ in range(10):
   135     81.5 MiB      0.0 MiB          10           x = Variable(np.random.randn(10000))
   136     81.5 MiB      0.6 MiB          10           y = square(square(square(x)))
"""



