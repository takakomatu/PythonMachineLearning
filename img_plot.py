#coding:utf-8
#!/usr/bin/env python
__author__ = 'HidetomoKataoka'

"""
##################################
画像plot用の関数ファイル
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



def draw_digit( data, size ):
	"""
	画像擬似plot関数（opencvを使用しない）
	input
		data: img_data( 1d_array )
		size: img_size( valiable )
	output
		matplotlibによる擬似画像を表示
	"""

	plt.style.use('ggplot')
	# 出力画像のサイズを指定
	plt.figure(figsize=(3, 3))
	# X は [0,1,・・・,26,27]の配列が28個できる --> array_size(28.28)
	# y は [0,0,・・・,0,0] ~ [27,27,・・・,27,27]の配列ができる --> array_size(28.28)
	X, Y = np.meshgrid(range(size),range(size))
	"""
	# debug code
	print np.shape(X)
	print X
	print np.shape(Y)
	print Y
	"""
	# 1次元756要素の配列を(28 * 28)の2次元配列に変換
	Z = data.reshape(size,size)
	Z = Z[::-1,:]
	"""
	# debug code
	listA = np.arange(150).reshape(3,50)
	print listA
	print listA[::-1,:]
	"""
	plt.xlim(0,27)
	plt.ylim(0,27)
	# 疑似カラープロットを描画
	plt.pcolor(X, Y, Z)
	# グレースケールで出力
	plt.gray()
	# x軸とy軸のラベル非表示
	plt.tick_params(labelbottom="off")
	plt.tick_params(labelleft="off")

	plt.show()





def draw_img( data, size, displayout ):
	"""
	画像出力関数（opencvを使用）
	input
		data: img_data( 1d_array )
		size: img_size( valiable )
		displayout:
			0 --> save_img
			1 --> plot_img
	output
		output.jpg or img_plot
	"""
	data *= 255
	# グレースケール表示のため、uint8に変換
	plt_data = data.astype(np.uint8)
	# 1次元756要素の配列を(28 * 28)の2次元配列に変換
	plt_data = data.reshape(size,size)

	# 画像を 表示 or 保存
	if displayout == 0:
		cv2.imwrite( "output.jpg", plt_data )
	elif displayout == 1:
		cv2.imshow( 'gray_img' , plt_data  )
		if cv2.waitKey(0):
			cv2.destroyAllWindows()
	else:
		print "plot_error"



def loss_accu_plot( loss_array, accuracy_array, file_name ):
	"""
	lossとaccuracyをplotする関数
	input
		loss_array: loss値配列
		accuracy_array: accuracy値の配列
	output
		loss値のaccuracy値のplot結果を出力(loss_accuracy.png)

	"""
	fig = plt.figure()
	
	ax1 = fig.add_subplot(211)
	ax1.plot(loss_array)
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('loss')
	
	ax2 = fig.add_subplot(212)
	ax2.plot(accuracy_array)
	ax2.set_xlabel('epoch')
	ax2.set_ylabel('accuracy')
	plt.savefig(file_name)
	
	#plt.show()
	print "save plot"




