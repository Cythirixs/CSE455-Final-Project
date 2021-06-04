from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

train_images = np.load("chara_train_images.npz")['arr_0']
train_labels = np.load("chara_train_labels.npz")['arr_0']
test_images = np.load("chara_test_images.npz")['arr_0']
test_labels = np.load("chara_test_labels.npz")['arr_0']

size = 48

if K.image_data_format() == "channels_first":
  train_images = train_images.reshape(train_images.shape[0], 1,size,size)
  test_images = test_images.reshape(test_images.shape[0], 1,size,size)
  shape = (1,size,size)
else:
  train_images = train_images.reshape(train_images.shape[0], size, size, 1)
  test_images = test_images.reshape(test_images.shape[0], size, size, 1)
  shape = (size,size,1)

print(train_images.shape[0])

datagen = ImageDataGenerator(rotation_range=15,zoom_range=0.2)
datagen.fit(train_images)
# model = keras.Sequential([
#   keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=shape),
#   keras.layers.MaxPooling2D(2,2),
#   keras.layers.Conv2D(64, (3,3), activation='relu'),
#   keras.layers.MaxPooling2D(2,2),
#   keras.layers.Flatten(),
#   keras.layers.Dropout(0.5),
#   keras.layers.Dense(2048, activation='relu'),
#   keras.layers.Dense(2, activation="softmax")
# ])

model = keras.Sequential([
  keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=shape),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(128, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(192, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(256, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(1024, activation='relu'),
  keras.layers.Dense(1024, activation='relu'),
  keras.layers.Dense(2, activation="softmax")
])
filepath = "saved-model.h5"

#model.summary()
#model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#model.fit_generator(datagen.flow(train_images,train_labels,shuffle=True),epochs=30,validation_data=(test_images,test_labels),callbacks = [keras.callbacks.EarlyStopping(patience=8,verbose=1,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,verbose=1),keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)])

chara_model = keras.models.load_model('saved-model.h5')
chara_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
#chara_model.fit_generator(datagen.flow(train_images,train_labels,shuffle=True),epochs=30,validation_data=(test_images,test_labels),callbacks = [keras.callbacks.EarlyStopping(patience=8,verbose=1,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,verbose=1),keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)])

test_loss, test_acc = chara_model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)

model.save("character.h5")
