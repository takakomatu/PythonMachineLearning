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
import cv2
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
batchsize = 10

# 学習回数
epoch = 10000

# 中間層のユニット数
#unit = 1000


if __name__ == "__main__":

	# 訓練用のデータ・ラベル、テスト用のデータ・ラベルをセット
	train_high_array, train_low_array, test_high_array, test_low_array = make_dataset.fsrcnn_dataset()
	print "train data and target",np.shape(train_high_array),np.shape(train_low_array)
	print "test data and target",np.shape(test_high_array),np.shape(test_low_array)

	# モデル生成
	model = network_model.SRCNN()
	#optimizer = optimizers.Adam()
	optimizer = optimizers.Adam()
	optimizer.setup(model)


	# モデルの学習
	loss_train_array     = []
	loss_test_array      = []
	N = len(train_low_array)
	E = len(test_low_array)
	
	# 学習（ミニバッチを使用）
	for j in range(epoch):
		sffindx = np.random.permutation( N )

		# training loop
		sum_train_loss       = 0
		for i in range(0, N, batchsize):
			x = Variable(train_low_array[sffindx[i:(i+batchsize) if (i+batchsize) < N else N]])
			y = Variable(train_high_array[sffindx[i:(i+batchsize) if (i+batchsize) < N else N]])
			model.zerograds()
			loss = model( x, y )
			loss.backward()
			optimizer.update()
			sum_train_loss += float(loss.data) * len(x)
		print "training"
		print 'epoch={}, train mean loss={}'.format( j+1, sum_train_loss / N )
		loss_train_array.append( sum_train_loss / N )
		
		"""
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
		"""
		if j%100 == 0:
			img_plot.loss_plot( loss_train_array, "../result/train_loss_plot" + str(j) + ".png" )
			xt = Variable(test_low_array)
			with chainer.using_config('train', False):
				test_data_result = model.fwd(xt)
			test_data_result = np.array(test_data_result.data)
			test_data_result = test_data_result.reshape( 2, 64*64 )
			test_data_result *= 255
			test_data_result = test_data_result.astype(np.uint8)
			for x in range(len(test_data_result)):
				temp = test_data_result[x]
				temp = temp.reshape(64,64)
				cv2.imwrite( "../result/sr_img/" + str(x) + "-" + str(j) + ".bmp", temp )
		print "###########################################################################"
	# 結果をplot
	img_plot.loss_plot( loss_train_array, "../result/train_loss_plot_final.png" )

	
	# テストデータの結果検証
	xt = Variable(test_low_array)
	with chainer.using_config('train', False):
		test_data_result = model.fwd(xt)

	print np.shape(test_data_result)

	test_data_result = np.array(test_data_result.data)
		
	test_data_result = test_data_result.reshape( 2, 64*64 )
	test_data_result *= 255
	print test_data_result[0]
	print np.shape(test_data_result),type(test_data_result)
	
	test_data_result = test_data_result.astype(np.uint8)

	for x in range(len(test_data_result)):
		temp = test_data_result[x]
		temp = temp.reshape(64,64)
		cv2.imwrite( "../result" + str(x) + "final" + ".bmp", temp )





