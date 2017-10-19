


L.lunear( in_unit_num, out_unit_num)



L.Convolution2D( input_channel_num, output_channel_num(filter_num), filter_size, stride, pad )
	stride: フィルターを動かしていくスライド幅
	        この値が大きいと、特徴量の取りこぼしが多くなる
	        	--> 値が小さい方が望ましい（計算量とのトレードオフ）
	pad: 画像のふちの外側に仮想的な画素を設ける
	　　　畳み込みの時の画像の縮小を避ける目的で利用する
	　　　畳み込みの時の画像の縮小を避ける目的で利用する
	　　  	--> 入力と出力の画像サイズを同じにしたい場合は、（filter_size / 2）
				※ 切り捨ての場合が多い



L.BatchNormalization( Just_before_channel_num )
	channel毎に平均と標準偏差が求められ、同じchannelに属するデータは全部同じの平均と標準偏差で正規化されている
	勾配消失や爆発を防ぐための手法（ReLU、重みの初期値を事前に学習、学習係数を下げる、dropout）の進化系
		--> ネットワークの学習プロセスを全体的に安定化させ、さらに学習速度を高める
	メリット: 
		1. 大きな学習係数が使える
			今までのdeeplearningでは、学習係数をあげるとパラメータのscaleの問題で、勾配消失や爆発が起きていた
			しかし、BatchNormalizationでは、伝搬中パラメータのscaleに影響を受けなくなる
				--> 学習を係数をあげることができ、学習の収束速度が向上する
		2. 正則化効果がある
			Dropoutは、過学習を抑える働きがあるが学習速度が遅くなる
			そこでBatchNormalizationを利用し、Dropoutを使わないことで学習速度を向上させることができる
		3. 初期値依存が少ない
			ニューラルネットワークの重みの初期値がそれほど性能に影響を与えなくなる



F.max_pooling_2d( input_img, ksize, stride )
	ksize: プーリング領域のサイズ
	       領域を大きくしすぎると精度が下がる
	stride: プーリング領域のストライド幅
	        通常は2以上にする
