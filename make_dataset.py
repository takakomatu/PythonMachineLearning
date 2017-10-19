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
import os


def fsrcnn_preprocess():
	
	print "FSRCNN_preprocess"

	dir_list = os.listdir("../train/train_original/")
	dir_list = dir_list[1:]
	print dir_list

	for x in dir_list:
		img_gray   = cv2.imread( "../train/train_original/" + x, cv2.IMREAD_GRAYSCALE )
		resize_img = cv2.resize( img_gray, (64,64) )
		cv2.imwrite( "../train/train_high/" + x, resize_img )
		resize_img = cv2.resize( resize_img, (32,32) )
		resize_img = cv2.resize( resize_img, (64,64) )
		cv2.imwrite( "../train/train_low/" + x, resize_img )

	dir_list = os.listdir("../test/test_original/")
	dir_list = dir_list[1:]
	print dir_list

	for x in dir_list:
		img_gray   = cv2.imread( "../test/test_original/" + x, cv2.IMREAD_GRAYSCALE )
		resize_img = cv2.resize( img_gray, (64,64) )
		cv2.imwrite( "../test/test_high/" + x, resize_img )
		resize_img = cv2.resize( resize_img, (32,32) )
		resize_img = cv2.resize( resize_img, (64,64) )
		cv2.imwrite( "../test/test_low/" + x, resize_img )


def fsrcnn_dataset():

	#fsrcnn_preprocess()

	print "FSRCNN_dataset"

	dir_list = os.listdir("../train/train_high/")
	print dir_list,"1"
	train_high_array = []
	for x in dir_list:
		img_gray   = cv2.imread( "../train/train_high/" + x, cv2.IMREAD_GRAYSCALE )
		img_gray   = img_gray.reshape( 64 * 64 )
		train_high_array.append( img_gray )


	dir_list = os.listdir("../train/train_low/")
	print dir_list,"2"
	train_low_array  = []
	for x in dir_list:
		img_gray   = cv2.imread( "../train/train_low/" + x, cv2.IMREAD_GRAYSCALE )
		img_gray   = img_gray.reshape( 64 * 64 )
		train_low_array.append( img_gray )


	dir_list = os.listdir("../test/test_high/")
	print dir_list,"3"
	dir_list = dir_list[1:]
	test_high_array = []
	for x in dir_list:
		img_gray   = cv2.imread( "../test/test_high/" + x, cv2.IMREAD_GRAYSCALE )
		img_gray   = img_gray.reshape( 64 * 64 )
		test_high_array.append( img_gray )


	dir_list = os.listdir("../test/test_low/")
	print dir_list,"4"
	dir_list = dir_list[1:]
	test_low_array  = []
	for x in dir_list:
		img_gray   = cv2.imread( "../test/test_low/" + x, cv2.IMREAD_GRAYSCALE )
		img_gray   = img_gray.reshape( 64 * 64 )
		test_low_array.append( img_gray )

	#print np.shape(train_high_array),np.shape(train_low_array)
	#print np.shape(test_high_array),np.shape(test_low_array)


		
	train_high_array  =  np.array(train_high_array).astype(np.float32)
	train_low_array   =  np.array(train_low_array).astype(np.float32)
	test_high_array   =  np.array(test_high_array).astype(np.float32)
	test_low_array    =  np.array(test_low_array).astype(np.float32)

	train_high_array /=  255
	train_low_array  /=  255
	test_high_array  /=  255
	test_low_array   /=  255

	train_high_array  =  train_high_array.reshape( len(train_high_array) ,1 ,64 ,64 )
	train_low_array   =  train_low_array.reshape(  len(train_low_array)  ,1 ,64 ,64 )
	test_high_array   =  test_high_array.reshape(  len(test_high_array)  ,1 ,64 ,64 )
	test_low_array    =  test_low_array.reshape(   len(test_low_array)   ,1 ,64 ,64 )

	#print np.shape(train_high_array),np.shape(train_low_array)
	#print np.shape(test_high_array),np.shape(test_low_array)

	
	return train_high_array, train_low_array, test_high_array, test_low_array



"""
# このファイル単体で回したい場合のみコメントアウト解除
if __name__ == "__main__":
	fsrcnn_dataset()
"""
