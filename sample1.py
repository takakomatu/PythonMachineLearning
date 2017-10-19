#coding:utf-8
#!/usr/bin/env python
__author__ = 'HidetomoKataoka'

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable 
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from sklearn.datasets import fetch_mldata
import sys
import matplotlib.pyplot as plt
import make_dataset
import img_plot
import network_model


"""
##################################
パラメータセットアップ
##################################
"""

# グラフプロットのスタイルをggplotに変更
plt.style.use('ggplot')

# 確率的勾配降下法を行う際のバッチサイズ
batchsize = 100

# 学習回数
epoch = 10

# 中間層のユニット数
#unit = 1000


if __name__ == "__main__":

	# 訓練用のデータ・ラベル、テスト用のデータ・ラベルをセット
	#train_data, train_target, test_data, test_tatget = make_dataset.mnist_dataset()
	train_data, train_target, test_data, test_tatget = make_dataset.mnist_dataset_cnn()
	print "train data and target",np.shape(train_data),np.shape(train_target)
	print "test data and target",np.shape(test_data),np.shape(test_tatget)
	"""
	# debug code	
	print "train data and target",np.shape(train_data),np.shape(train_target)
	print "test data and target",np.shape(test_data),np.shape(test_tatget)
	"""

	# モデル生成
	model = network_model.mnist_model()
	#model = network_model.CNN()
	optimizer = optimizers.Adam()
	optimizer.setup(model)


	# モデルの学習
	loss_train_array     = []
	accuracy_train_array = []
	loss_test_array      = []
	accuracy_test_array  = []
	N = len(train_data)
	E = len(test_data)
	
	# 学習（ミニバッチを使用）
	for j in range(epoch):
		sffindx = np.random.permutation( N )

		# training loop
		sum_train_loss       = 0
		sum_train_accuracy   = 0
		for i in range(0, N, batchsize):
			x = Variable(train_data[sffindx[i:(i+batchsize) if (i+batchsize) < N else N]])
			y = Variable(train_target[sffindx[i:(i+batchsize) if (i+batchsize) < N else N]])
			model.zerograds()
			loss, accuracy = model( x, y )
			loss.backward()
			optimizer.update()
			sum_train_loss += float(loss.data) * len(x)
			sum_train_accuracy += float(accuracy.data) * len(x)
		print "training"
		print 'epoch={}, train mean loss={}, accuracy={}'.format( j+1, sum_train_loss / N, sum_train_accuracy / N)
		loss_train_array.append( sum_train_loss / N )
		accuracy_train_array.append( sum_train_accuracy / N )

		# test evaluation loop
		sum_test_loss        = 0
		sum_test_accuracy    = 0
		for i in range(0, E, batchsize):
			x = Variable(test_data[i:(i+batchsize) if (i+batchsize) < E else E])
			y = Variable(test_tatget[i:(i+batchsize) if (i+batchsize) < E else E])
			loss, accuracy = model( x, y )
			sum_test_loss += float(loss.data) * len(x)
			sum_test_accuracy += float(accuracy.data) * len(x)
		print "evaluation"
		print 'epoch={}, train mean loss={}, accuracy={}'.format( j+1, sum_test_loss / E, sum_test_accuracy / E)
		loss_test_array.append( sum_test_loss / E )
		accuracy_test_array.append( sum_test_accuracy / E )
		print "###########################################################################"
	# 結果をplot
	img_plot.loss_accu_plot( loss_train_array, accuracy_train_array, "train_loss_accuracy_plot.png" )
	img_plot.loss_accu_plot( loss_test_array, accuracy_test_array, "test_loss_accuracy_plot.png" )


	# テストデータの結果検証
	xt = Variable(test_data)
	with chainer.using_config('train', False):
		test_data_result = model.fwd(xt)

	ans = test_data_result.data
	nrow, ncol = ans.shape
	ok = 0
	for i in range(nrow):
		cls = np.argmax(ans[i,:])
		print ans[i,:], cls, test_tatget[i]
		if cls == test_tatget[i]:
			ok += 1
	print ok, "/", nrow, " = ", (ok * 1.0)/nrow







