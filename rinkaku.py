# coding: utf-8
#rinkaku9では読み込んだ画像を処理し、処理結果をlistに格納して出力する。(学習用データではなく、testデータを読み込むことを想定している。)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier


for root, dirs, files in os.walk("allImage_files"):
	for f in files:
		if os.path.splitext(f)[1] == ".png":#rootとextに"."で分離 os.path.splitext(f)[1]は後者=(ext)
			img_src = cv2.imread(os.path.join(root, f),1)
			img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)#グレースケールに変換
			thresh = 225#ここの閾値で調整   
			max_pixel = 255
			ret,img_dst = cv2.threshold(img_gray,thresh,max_pixel,cv2.THRESH_BINARY)

			# img = cv2.imread('20170209_M6-15.png',0)
			ret,thresh = cv2.threshold(img_dst,127,255,0)

			#輪郭抽出

			imgEdge,contours,hierarchy = cv2.findContours(thresh, 1, 2) #1 = cv2.RETR_TREE?, 2= cv2.CHAIN_APPROX_SIMPLE?
			# plt.imshow(imgEdge) 
			# plt.show()
			answerlist = []
			for cnt in contours:
				datalist = []
				
				#1,角度考慮ありの矩形面積
				rect = cv2.minAreaRect(cnt)
				# print(rect) #中心のx,y, width, length, angleを返す
				#print (rect[1][0]*rect[1][1])#面積を計算(width,lengthの積)しているハズ
				# print(rect)
				element0 = rect[0][0]#中心のx座標
				
				element1 = rect[1][0]*rect[1][1]
				box = cv2.boxPoints(rect)
				box = np.int0(box)
				img = cv2.drawContours(img_dst,[box],0,(0,0,255),2)
				#角度考慮ありの矩形面積の描画
				# plt.imshow(img) 
				# plt.show()


				#2,角度考慮ありの矩形内部のネジ領域ドット面積
				M = cv2.moments(cnt)
				#print M['m00']
				element2 = M['m00']
				# area = cv2.contourArea(cnt)と同値
				

				#3,角度考慮ありの矩形の辺の長さ　(長い方)
				width = rect[1][0]
				length = rect[1][1]

				if width > length:
					element3 = width
				else:
					element3 = length

				#3b,角度考慮ありの矩形の辺の長さ　(短い方)
				

				if width > length:
					element3b = length
				else:
					element3b = width
								


				#4,輪郭の長さ (周囲長(arc length))
				perimeter = cv2.arcLength(cnt,True) #閉じているもののみ計算
				#print(perimeter)
				element4 = perimeter

				#5,アスペクト比(物体を囲む外接長方形の縦幅(height)に対する横幅(width)の比)
				
				# if (width) > (length)&(width != 0)&(length != 0):
				#  	print (width / length)
				# else:
				# 	print (length / width)


				#以下はおそらく回転を考慮できていない
				# x, y, w, h = cv2.boundingRect(cnt)
				# aspect_ratio = float(w)/h

				#6,Extentは(外接矩形の面積に対する輪郭が占める面積の比)
				# if (width != 0)&(length != 0):
					#print (float(M['m00'] / (rect[1][0]*rect[1][1])))
				if (rect[1][0] != 0) and (rect[1][1] != 0):
					element5 = (M['m00'] / (rect[1][0]*rect[1][1]))
				
				#8,等価直径:輪郭の面積と同じ面積を持つ円の直径
				area = cv2.contourArea(cnt)
				equi_diameter = np.sqrt(4*area/np.pi)
				#print (equi_diameter)
				element6 = equi_diameter

				if (element2 > 500) and (element1 < 1000000): #面積がある程度以上の場合のみを取得
					# datalist.append(element0)#中心のx座標　該当する画像を探すために記載した。
					datalist.append(element1)#角度考慮ありの矩形面積
					datalist.append(element2)#輪郭内面積
					datalist.append(element3)#矩形の辺の長さ(longer)
					datalist.append(element3b)#矩形の辺の長さ(shorter)
					datalist.append(element4)#輪郭の長さ
					datalist.append(element5)#外接矩形の面積に対する輪郭が占める面積の比
					datalist.append(element6)#等価直径
					# dataResult =  (str(datalist) + ",")#listの末尾に","を付加して出力 ただ、文字として認識されるため、後のExel処理に影響を与える
					answerlist.append(datalist)
			print (answerlist)#answerListにはdatalistを順次appendしていった結果がlist化して含まれる。
					# print ((element0), (element1))
					


			# plt.imshow(img) 
			# plt.show()






#以下では学習データを読み込ませる　#学習データ:既存の37個分を使用	
X = [
[38492.996467655525, 23101.0, 261.61993408203125, 147.13327026367188, 911.920914888382, 0.6001351445687284, 171.50249771629274],
[37793.50158563675, 23356.5, 260.1219787597656, 145.2914581298828, 779.4528781175613, 0.6180030698419469, 172.44830943389044],
[36325.685063452926, 20851.0, 252.21426391601562, 144.02708435058594, 944.4894567728043, 0.5740015629045377, 162.93654515569207],
[37163.83518720092, 21382.0, 256.1646423339844, 145.0779266357422, 950.4305790662766, 0.575344280058692, 164.99820588578305],
[38451.83687930927, 22363.5, 258.28619384765625, 148.87298583984375, 969.3595106601715, 0.5815977028664054, 168.74268149666466],
[40152.124618664384, 24805.5, 264.7794189453125, 151.6436767578125, 793.1097319126129, 0.6177879809744705, 177.7170321801714],
[34970.77629195037, 21146.0, 253.3717803955078, 138.02159118652344, 882.1463084220886, 0.6046763109707525, 164.08511027198583],
[32636.254372875206, 18658.0, 241.53807067871094, 135.11846923828125, 904.430581331253, 0.5716955072977714, 154.13015092988348],
[33328.356498662615, 20707.0, 251.41319274902344, 132.56407165527344, 776.6660803556442, 0.621302763634052, 162.37293879471116],
[34142.73719156394, 19903.5, 247.32223510742188, 138.0496063232422, 900.6488342285156, 0.5829497467741924, 159.19146735499459],
[32765.29026213102, 18570.5, 243.0020751953125, 134.83543395996094, 857.1341171264648, 0.5667735537036623, 153.76831587002681],
[35668.71654106304, 20509.0, 252.44094848632812, 141.2952880859375, 906.4305793046951, 0.5749856453732879, 161.59477040725497],
[32979.99678942561, 21001.0, 240.41629028320312, 137.1787109375, 683.9209150075912, 0.6367799285757832, 163.52156946098319],
[34649.996400146745, 22244.5, 247.4873504638672, 140.00714111328125, 710.7909752130508, 0.6419769786731, 168.29312835900737],
[33806.07631713874, 20487.0, 246.3999786376953, 137.1999969482422, 767.0924869775772, 0.6060153153477223, 161.50807581353101],
[33037.201858313056, 19837.0, 245.23988342285156, 134.7138214111328, 842.7493433952332, 0.6004443138094782, 158.92530587955909],
[35155.40341897169, 21472.5, 249.9324188232422, 140.65963745117188, 826.3056910037994, 0.6107880414312731, 165.34701728282184],
[33042.994799792534, 21267.0, 244.65892028808594, 135.05738830566406, 708.5483348369598, 0.6436159957309174, 164.55389815462502],
[57990.8442786918, 37270.0, 302.5889587402344, 191.64891052246094, 1021.8864245414734, 0.6426876598120941, 217.83855910347808],
[60382.54479042487, 39208.0, 306.4132385253906, 197.0624542236328, 942.0630512237549, 0.6493267240736993, 223.43047256356118],
[56641.96188222617, 34767.0, 292.49395751953125, 193.6517333984375, 1136.0285576581955, 0.6138028917905407, 210.39657614088543],
[53703.0081812772, 33063.0, 286.3782653808594, 187.5247344970703, 1103.8864263296127, 0.6156638355973316, 205.1758247639782],
[56220.140731299296, 36881.5, 299.8615417480469, 187.48699951171875, 924.3056910037994, 0.6560193468079858, 216.70021751061975],
[53921.53815674689, 33376.0, 289.0863952636719, 186.52395629882812, 1123.283392906189, 0.6189734406866851, 206.14471384219578],
[55777.886501706205, 36178.0, 296.0669250488281, 188.39620971679688, 879.7787801027298, 0.6486082974637869, 214.62353144384872],
[53514.78265613504, 34842.5, 296.0255126953125, 180.77760314941406, 988.4478244781494, 0.6510817809703949, 210.62490080100906],
[49931.10276951408, 31658.5, 290.1215515136719, 172.1040802001953, 1063.952437877655, 0.6340436770671407, 200.77065056177446],
[49450.157161827665, 30560.0, 279.1709289550781, 177.13218688964844, 1125.54327917099, 0.6179960136424066, 197.256686799476],
[50339.093250709586, 33229.0, 290.59503173828125, 173.22764587402344, 964.3473210334778, 0.6601032687360057, 205.69024486349545],
[50706.995118571445, 31805.0, 283.8622741699219, 178.63238525390625, 1067.6021527051926, 0.6272310146879797, 201.23464840901988],
[52439.98007202335, 32531.0, 284.99993896484375, 183.99996948242188, 1029.945298075676, 0.6203472990516112, 203.51844051529969],
[50307.37884423137, 33437.5, 277.339599609375, 181.3927001953125, 812.0458083152771, 0.6646639274038465, 206.33455182562616],
[52273.321352759376, 35026.5, 287.8326416015625, 181.61012268066406, 846.1290677785873, 0.6700645586230966, 211.18031374554349],
[49944.48333972506, 31552.0, 282.35833740234375, 176.88333129882812, 1014.371707201004, 0.63174144350201, 200.43266728625815],
[49379.64312472567, 33597.5, 283.4884033203125, 174.18576049804688, 845.0163662433624, 0.6803917135475784, 206.82762292363086],
[51840.70351465605, 33028.5, 286.0963134765625, 181.20018005371094, 1004.4722092151642, 0.6371151963758055, 205.06875018706609],
[50428.31185946101, 32138.5, 282.0267028808594, 178.80686950683594, 936.187940955162, 0.637310645844481, 202.28694744958466]
]

y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

model = RandomForestClassifier()
model.fit(X, y)

#データが一個の場合は二重括弧。test_data = [[36000, 21176, 255, 144, 940, 0.57, 164]]
test_data = answerlist
output = model.predict(test_data)

print (output)
# plt.imshow(img) 
# plt.show()
