#coding:utf-8

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


"""
Chainerのお作法
・データは配列からChainerのVariableという型（クラス）のオブジェクトに変換して使用
	array --> Variable
・偏微分したい式にbackward()を行い、変数ごとにgradすると偏微分値が求まる
"""

x1 = Variable(np.array([1],dtype=np.float32))
print x1
print x1.data
x2 = Variable(np.array([2],dtype=np.float32))
x3 = Variable(np.array([3],dtype=np.float32))

z = (x1 -2*x2-1)**2 + (x2*x3-1)**2 + 1

z.backward()
print x1.grad
print x2.grad
print x3.grad



x = Variable(np.array([-1], dtype=np.float32))
print F.sin(x)
print F.sin(x).data
print F.sigmoid(x)
print F.sigmoid(x).data


print "*****************************************"

"""
・chainer.functionの中に活性化関数や損失関数などが定義されている
・変数が多次元の場合は、関数の傾きの次元をあらかじめ設定しておく必要がある
	z.grad = np.ones(z.size,dtype=np.float32)  -->  z.backward()  -->  x.grad
・chainer.linksの中にはパラメータありの関数を提供している
	chainer.functionの中の関数はパラメータなし
	自分が考えたモデル（合成関数）をlinks内の関数とfuncions内の関数	を組み合わせて表現する
"""

x = Variable(np.array([-0.5], dtype=np.float32))
z = F.cos(x)
print z.data
z.backward()
print x.grad
print ((-1)*F.sin(x).data)


# 変数が多次元な場合
x = Variable(np.array([-1,0,1], dtype=np.float32))
print x.data

z = F.sin(x)
print z
z.grad = np.ones(z.size,dtype=np.float32)
print z.grad
z.backward()
print z
print x.grad


h = L.Linear(3,4)
print h
print h.W.data
print np.shape(h.W)
print h.b.data

print "*****************************************"

"""
・L.Linearの場合　y = Wx + b
	W --> 重み行列( l+1層のノード数, l層のノード数)・初期値はランダムな値が入っている
	b --> バイアス（ l+1層のノード数 ）・初期値は全て0
"""

x = Variable(np.array(range(6)).astype(np.float32).reshape(2,3))
print x
print x.data

h = L.Linear(3,4)
y = h(x)
print y.data


w = h.W.data
x0 = x.data
x1 = x0.dot(w.T) + h.b.data
print x1


print "*****************************************"

"""
・chainクラスはモデルを定義するもの（合成関数全体を表現する）
y = w2( σ( W1X + b ) ) + b2
・__init__
	コンストラクタでモデルを構成するレイヤーを定義する
	この際、親クラス（MyChain）のコンストラクタにsuperを用いてキーワード引数としてモデルを構成するLinkオブジェクトを渡すことで
	Optimizerから捕捉可能な最適化対象のパラメータを持つレイヤをモデルに追加することができる
・データを受け取る()アクセサで呼ばれる__call__メソッドに、Forward計算を記述する
"""

# このクラスを定義することで、訓練データを与えれば、損失関数をパラメータで微分（偏微分）した値がもとまる
class MyChain(Chain):
	#モデル（パラメータを持つ層）の定義
	def __init__(self):
		super(MyChain, self).__init__(
			l1 = L.Linear(4,3),
			l2 = L.Linear(3,3),
		)

	# モデルを定義するという意味ではfoward計算を記述する方がスマート
	# しかし、モデルのfoward計算は損失関数の計算の中で必要とされるので、foward計算は別のクラスメソッドとしておけば良い
	# x --> 損失関数への入力
	# y --> 教師データ
	# mean_squared_error --> 自乗誤差(MSE)
	def __call__(self,x,y):
		fv = self.fwd(x,y)
		loss = F.mean_squared_error(fv,y)
		return loss

	# foward計算
	def fwd(self,x,y):
		return F.sigmoid(self.l1(x))


print "*****************************************"

"""
optimizers
・偏微分値からパラメータを更新するプログラムを書くのは面倒 --> Optimizerを利用することでパラメータを更新
"""

# モデルを生成
# 最適化アルゴリズムをセット（今回はSGD（確率的勾配降下法）を利用） --> 通常はAdamを利用する
# アルゴリズムをモデルにセット（適用）
model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)


# 一つの訓練データんおバッチ(x,y)を与えると、パラメータが一回更新される場合
# 勾配の初期化
# fowardで計算し、誤差を求める
# backwardの計算（勾配の計算）
# パラメータを更新
"""
# xとyがないため、コメントアウト中
model.zerograds()
loss = model(x,y)
loss.backward()
optimizer.update()
"""




