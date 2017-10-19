#coding:utf-8
#!/usr/bin/env python

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn import datasets
import sys


"""
Irisデータ
    150個のデータ（あやめのデータ）
    各データは(花びらの長さ、花びらの幅、がく片の長さ、萼片の幅)  -->  4次元ベクトル
    ラベルは3種類

    奇数番目を訓練データ（75 data）
    偶数番目をテストデータ（75 data）
"""





# データをセットする

iris = datasets.load_iris()
# 4次元の特等ベクトル（150 data）
X = iris.data.astype(np.float32)
# 3種類のラベル（150 data）
Y = iris.target
# データ数
N = Y.size

Y2 = np.zeros(3 * N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0

# 0 ~ 149の値が順に入っている
index = np.arange(N)
# 訓練データ（奇数番目）
xtrain = X[index[index % 2 != 0],:]
ytrain = Y2[index[index % 2 != 0],:]
# テストデータ（偶数番目）
xtest = X[index[index % 2 == 0],:]
yans = Y[index[index % 2 == 0]]



print "*****************************************"


# ネットワークモデル
"""
"""
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1=L.Linear(4,6),
            l2=L.Linear(6,3),
        )
        
    def __call__(self,x,y):
        fv = self.fwd(x)
        return F.mean_squared_error(fv, y)

    def fwd(self,x):
         h1 = F.sigmoid(self.l1(x))
         h2 = self.l2(h1)
         #h3 = F.softmax(h2)
         return h2


print "*****************************************"


# モデル生成

model = IrisChain()
# 最適化アルゴリズム（SGD）
optimizer = optimizers.SGD()
# 最適化アルゴリズムをモデルにセット
optimizer.setup(model)


print "*****************************************"

# モデル学習
"""
誤差累積を使用
パッチ学習と似た学習方法だが、誤差の累積から勾配を求め、パラメータを更新する
    --> 各データに対してそれぞれ勾配を求め、それらの総和をとったものを累積された誤差の勾配として使用

ミニバッチ --> 各データから得た勾配の平均を用いてパラメータを更新
誤差累積　 --> 各データから得た勾配の総和を用いてパラメータを更新

今回はミニバッチの誤差を累積させるコードを書く
"""

n = 75    
bs = 25   
for j in range(2000):
    accum_loss = None    
    sffindx = np.random.permutation(n)
    for i in range(0, n, bs):
        x = Variable(xtrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        y = Variable(ytrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        model.zerograds()
        loss = model(x,y)
        accum_loss = loss if accum_loss is None else accum_loss + loss
    accum_loss.backward()
    optimizer.update()


print "*****************************************"


# varidation
"""
xt = Variable(xtest, volatile='on')
これはver1は使えたがver2では使用できない
"""

xt = Variable(xtest)
"""
# この書き方でもOK
with chainer.no_backprop_mode():
    yy = model.fwd(xt)
"""
with chainer.using_config('train', False):
    yy = model.fwd(xt)

print "*****************************************"


# modelヵら出力された推定結果と真値を比べて正解率を出力

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print ans[i,:], cls            
    if cls == yans[i]:
        ok += 1
        
print ok, "/", nrow, " = ", (ok * 1.0)/nrow

print "*****************************************"





