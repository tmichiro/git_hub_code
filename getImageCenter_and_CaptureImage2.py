# coding:utf-8

#1画像からネジを検出し、画像の中心座標を求める
#2画像の中心座標から一定の幅を持つ四角形で画像をsaveする

#以下の3か所に調べたいネジを含むファイル名、出力ファイル名を記載する
#image = cv2.imread("imageCopy_M8-16_15b.png",0)
#image3 = cv2.imread("imageCopy_M8-16_15b.png",1)
#cv2.imwrite('imageCopy_M8-16_15b.png', image3[a:b,c:d])  



import cv2
import matplotlib.pyplot as plt
import numpy as np


#個別ファイルを指定する場合は以下
image = cv2.imread("imageCopy_M6-15_1b.png",0)
# plt.imshow(image)
# plt.show()


#二値化
thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] #閾値0, 最大値255 = 白 

# plt.imshow(thresh)
# plt.show()

imgEdge,contours,hierarchy = cv2.findContours(thresh, 1, 2) #1 = cv2.RETR_TREE?, 2= cv2.CHAIN_APPROX_SIMPLE?
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	if h < 400: #画像の外枠の情報を取得しないようにする
		img = cv2.rectangle(image, (x,y),(x+w,y+h), (0,0,255), 2)
		plt.imshow(img)
		plt.show()

		print ("X: {}".format(x + w/2) )
		print ("Y: {}".format(y + h/2) )
		print ("X-200 {}".format((x + w/2)-200))
		print ("X+200 {}".format((x + w/2)+200))
		print ("Y-200 {}".format((y + h/2)-200))
		print ("Y+200 {}".format((y + h/2)+200))

		centerX = (x + w/2)
		centerY = (y + h/2)

		a = centerY - 200
		b = centerY + 200
		c = centerX - 200
		d = centerX + 200

		#401pixelで画像を取得　(ネジの中心(外接矩形中心)= (centerX,centerY) )
		img2 = cv2.rectangle(image, (centerY-200, centerY + 200),(centerX - 200, centerX + 200), (0,0,255), 2)
		plt.imshow(img2) #ここは外接矩形のチェックを行う際には実行する
		plt.show()


#二値化していない元の画像に対して枠を作成、表示する
image3 = cv2.imread("imageCopy_M6-15_1b.png",1)
# img3 = cv2.rectangle(image3, (centerY-200, centerY + 200),(centerX-200, centerX + 200), (0,0,255), 2)
# cv2.imwrite('demo.jpg', image3[y:y + hight, x:x + width])
cv2.imwrite('imageCopy_M6-15_1bcut.png', image3[a:b,c:d])  #[y:y + hight, x:x + width]で範囲を指定できる
#上記の値はcntの中から最適と判断したものを仕様


# plt.imshow(img4)
# plt.show()
