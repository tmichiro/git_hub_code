# coding: utf-8
# makedata4_for_deep.py　(neji4.npyを出力)とともに用いる


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np


# root_dir = "neji_pics" #ここはscript内では読まれない
categories = ["large", "mini"] #この行はlen(categories)にのみ使用される
nb_classes = len(categories)
image_size = 50

# データをload
def main():
	X_train, X_test, y_train, y_test = np.load("neji4.npy")#square_binary_neji40_png,makedata4_for_deep2.pyと連動
	X_train = X_train.astype("float")/256
	X_test = X_test.astype ("float")/256
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)
	model = model_train(X_train, y_train)
	model_eval(model, X_test, y_test)

#modelの構築	
def build_model(in_shape):
	model = Sequential()
	model.add(Conv2D(32,(3,3), padding = 'same', input_shape = in_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64,(3,3), padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64,(3,3)))
	model.add(MaxPooling2D(pool_size= (2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	model.compile(loss = 'binary_crossentropy', optimizer= 'rmsprop', metrics = ['accuracy']) #binary_crossentropy
	return model


#modelの訓練
def  model_train(X, y):
	model = build_model(X.shape[1:])
	model.fit(X, y, batch_size = 32, epochs = 30)

	#modelの保存
	hdf5_file = "neji_model.hdf5"  # 参考https://keras.io/ja/getting-started/faq/   ...model.load_weights('my_model_weights.h5')
	model.save_weights(hdf5_file)
	return model


#modelの評価
def model_eval(model, X,y):
	score = model.evaluate(X, y)
	print ("loss" , score[0])
	print ("accuracy= ", score[1])


if __name__ == '__main__':
	main() 