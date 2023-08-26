import weakref

import numpy as np


class Variable:

    # 左項にndarrayが来たときにも，Variableの__radd__がよばれるようにしたい
    __array_priority__ = 200

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
    
    # def __mul__(self, other):
    #     return mul(self, other)
    
    # def __add__(self, other):
    #     return add(self, other)

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

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
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
    
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)
    
class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0
    
def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx
    
def square(x: Variable):
    return Square()(x)
    
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx
    
def exp(x: Variable):
    return Exp()(x)
    
class Neg(Function):
    def forwar(self, x):
        return -x
    def backward(self, gy):
        return -gy
    
def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    def backward(self, gy):
        return gy, -gy
    
def sub(x0, x1):
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1
    
def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

class Pow(Function):
    def __init__(self, c):
        self.c = c
    def forward(self, x):
        y = x ** self.c
        return y
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx
    
def pow(x, c):
    return Pow(c)(x)


# __mul__()をクラス内に実装しない書き方
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow



if __name__ == '__main__':
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))

    # y = add(mul(a, b), c)
    # print(y)

    # y.backward()
    # print(y.grad)
    # print(a.grad)
    # print(b.grad)

    # # 掛け算演算子のオーバーロード
    # y = a * b
    # print(y)

    x = Variable(np.array(0.5))
    y = x + np.array(3)
    print(y)
    z = 3.0 * x + 1.0
    print(z)
    print(3 + x)

    print(np.array(3) + x)

    x = Variable(np.array(2.0))
    print(x ** 3)



