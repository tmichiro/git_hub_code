# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# for angle in range(0, 360, 60):       の60は60度ごと回転という意味

# 画像読み込み(alphaチャンネル有り)
src_mat = cv2.imread("binarygrayM6-15_3bcut.png", cv2.IMREAD_UNCHANGED) #保存時にネジの大きさの区別に注意
#回転させる前の大元の画像データを読み込む
#binarygrayM8-16_1bcut.pngのような、「中心をとってあり、なおかつ余白に余裕があるネジの単一画像」を
#templateとして回転させると綺麗な回転画像が得られる。
#名前の変更の注意点...以下
#大ネジ#binarygrayM8-16_1bcut.png...Large_rec
#小ネジbinarygrayM6-15_3bcut.pngS...mall_rec


#imread(filename[,flag])のflag相当
#cv2.IMREAD_UNCHANGED = -1...alphaチャネル込みでコピー	
# grayscale = 0
# native = 1

#alphaチャネルで背景透過ができる??
#print(src_mat.shape) 

# 画像サイズの取得(横, 縦)
size = tuple([src_mat.shape[1], src_mat.shape[0]])

# # dst 画像用意np.zeros((size[1],size[0]))
dst_mat = np.zeros((size[1],size[0], 4), np.uint8)

# 画像の中心位置(x, y)
center = tuple([int(size[0]/2), int(size[1]/2)])


# 拡大比率
scale = 1.0

# 回転角度（正の値は反時計回り）
for angle in range(0, 360, 3): #回転角度はここで調整

	# 回転変換行列の算出
	rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
	# アフィン変換
	img_dst = cv2.warpAffine(src_mat, rotation_matrix, size, dst_mat,
	                         flags=cv2.INTER_LINEAR,
	                         borderMode=cv2.BORDER_TRANSPARENT)
	cv2.imwrite("/Users/tabataimac27/rotation_neji1/Small_dst_angle_{}.png".format(angle), img_dst) #ネジサイズで名前変更
	#rotation_neji1フォルダに一回結果を出力


for root, dirs, files in os.walk("rotation_neji1"):
	for f in files:
		if os.path.splitext(f)[1] == ".png":#rootとextに"."で分離 os.path.splitext(f)[1]は後者=(ext)
			img_src = cv2.imread(os.path.join(root, f),1)
			img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)#グレースケールに変換

			# image1 = cv2.imread("/Users/tabataimac27/rotation_neji/dst_angle_0.png",0)
			
			#二値化
			thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #閾値0, 最大値255 = 白 
			# plt.imshow(thresh)
			# plt.show()

			image2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)			

			#cv2.EXTERNALを用いると、最も外側のみを検出するので、今回は不適
			detect_count = 0

			for i in range(0, len(contours)):
				area = cv2.contourArea(contours[i]) #i番目の矩形の面積 面積はトリミングで使用する
				if 10000 < area and area < 100000: #目的のネジを取得するための変数の設定　大は30000くらいが欲しいネジ　小ネジは10000以上20000以下くらい？	
					rect = contours[i]
					x,y,w,h = cv2.boundingRect(rect)
					# print "{},{},{},{}".format(x,y,w,h)
					imageCopied = img_src[y:y+h, x:x+w]
					# plt.imshow(imageCopied)
					# plt.show()
					cv2.imwrite("/Users/tabataimac27/rotation_neji2/{}.png".format("Small_rec" + os.path.splitext(f)[0]),imageCopied) #ネジサイズで名前変更
					#名前の付け方注意.format(i)だとfor構文が回らない。
					#読み込んだ写真の名前が異なることを利用して保存名を作成	
					#ネジの種類ごとに名前を変えること
