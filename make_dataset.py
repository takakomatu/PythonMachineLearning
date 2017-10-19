#coding:utf-8
#!/usr/bin/env python
__author__ = 'HidetomoKataoka'

"""
##################################
訓練とテストに使うデータセットを作成
##################################
"""


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
import cv2
import img_plot
from sklearn.cross_validation import train_test_split

def mnist_dataset():
	"""
	MNISTの手書き数字のデータをダウンロード
	28x28ピクセル、70000サンプル
	各ピクセルは0から255の値を取る（グレースケール画像）
	mnistのデータサイズは(70000,784)
	画像一枚は一次元になっている
	"""

	print "fetch MNIST dataset"
	mnist = fetch_mldata("MNIST original")
	"""
	# debug code
	print type(mnist)
	print mnist.data
	print np.shape(mnist.data)
	print mnist.data[0]
	print np.shape(mnist.data[0])
	"""

	# chainerのお作法で、dataはfloat32にする
	mnist.data   = mnist.data.astype(np.float32)
	# 0 ~ 1に正規化
	mnist.data  /= 255
	# chainerのお作法で、targetはint32にする
	mnist.target = mnist.target.astype(np.int32)
	"""
	# debug code
	print mnist.data[0]
	print mnist.target
	print np.shape(mnist.target)
	"""

	"""
	# debug code
	#img_plot.draw_digit(mnist.data[0],28)
	#img_plot.draw_digit(mnist.data[1000],28)
	#img_plot.draw_digit(mnist.data[5000],28)
	#img_plot.draw_img(mnist.data[0],28,0)
	#img_plot.draw_img(mnist.data[0],28,1)
	"""

	# 70000データのうち60000データを訓練データに、10000データをテストに使用する
	N = 60000
	train_data, test_data     = np.split(mnist.data,   [N])
	train_target, test_tatget = np.split(mnist.target, [N])
	"""
	# debug code
	print "all_data",np.shape(mnist.data),np.shape(mnist.target)
	print "train data and target",np.shape(train_data),np.shape(train_target)
	print "test data and target",np.shape(test_data),np.shape(test_tatget)
	"""

	return train_data, train_target, test_data, test_tatget

def mnist_dataset_cnn():
	
	print "fetch MNIST dataset"
	mnist = fetch_mldata("MNIST original")
	
	# chainerのお作法で、dataはfloat32にする
	mnist.data   = mnist.data.astype(np.float32)
	# 0 ~ 1に正規化
	mnist.data  /= 255
	# chainerのお作法で、targetはint32にする
	mnist.target = mnist.target.astype(np.int32)

	# 70000枚, 1ch, tate, image_size(28 x 28)
	mnist.data = mnist.data.reshape(70000,1,28,28)

	# 70000データのうち90%を訓練データに、10%をテストに使用する
	train_data,test_data,train_target,test_tatget = train_test_split( mnist.data, mnist.target, test_size = 0.1 )
	#"""
	# debug code
	print "all_data",np.shape(mnist.data),np.shape(mnist.target)
	print "train data and target",np.shape(train_data),np.shape(train_target)
	print "test data and target",np.shape(test_data),np.shape(test_tatget)
	#"""

	return train_data, train_target, test_data, test_tatget


"""
# このファイル単体で回したい場合のみコメントアウト解除
if __name__ == "__main__":
	mnist_dataset()
"""
	
