import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None # 逆伝搬で受け取った勾配を保持
        self.creator = None # 自身を生み出した関数を記憶

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # 勾配の初期化を省略するためのコード
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # 再帰を使わないstep8
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # リストの末尾の取得
            gys = [output.grad for output in f.outputs]
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
                    funcs.append(x.creator)

        # 再帰を使うstep7 
        # f = self.creator # 自身を生み出した関数
        # if f is not None:
        #     x = f.input # その関数の入力
        #     x.grad = f.backward(self.grad) # 勾配はその関数の微分に自身の勾配をかけたもの
        #     x.backward() # これを再帰的に呼ぶことで自動化する
    
    def cleargrad(self):
        # step14: 微分のリセット
        # Variable.backwardの変更に伴ってこちらのメソッドを追加
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
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
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

if __name__ == '__main__':

    # z = x^2 + y^2 の微分
    x = Variable(np.array(2))
    y = Variable(np.array(3))
    z = add(square(x), square(y))
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)

