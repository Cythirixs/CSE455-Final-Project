from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

train_images = np.load("kanji_train_images.npz")['arr_0']
train_labels = np.load("kanji_train_labels.npz")['arr_0']
test_images = np.load("kanji_test_images.npz")['arr_0']
test_labels = np.load("kanji_test_labels.npz")['arr_0']

if K.image_data_format() == "channels_first":
  train_images = train_images.reshape(train_images.shape[0], 1,64,64)
  test_images = test_images.reshape(test_images.shape[0], 1,64,64)
  shape = (1,64,64)
else:
  train_images = train_images.reshape(train_images.shape[0], 64, 64, 1)
  test_images = test_images.reshape(test_images.shape[0], 64, 64, 1)
  shape = (64,64,1)
  
datagen = ImageDataGenerator(rotation_range=10,zoom_range=0.2)
datagen.fit(train_images)
model = keras.Sequential([
  keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=shape),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Conv2D(64, (3,3), activation='relu'),
  keras.layers.MaxPooling2D(2,2),
  keras.layers.Flatten(),
  keras.layers.Dropout(0.5),
  keras.layers.Dense(2048, activation='relu'),
  keras.layers.Dense(879, activation="softmax")
])

# model = keras.Sequential([
#   keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=shape),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(128, (3,3), activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(512, (3,3), activation='relu'),
#   keras.layers.Conv2D(512, (3,3), activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Flatten(),
#   keras.layers.Dropout(0.5),
#   keras.layers.Dense(4096, activation='relu'),
#   keras.layers.Dense(4096, activation='relu'),
#   keras.layers.Dense(879, activation="softmax")
# ])

#model.summary()

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
              
filepath = "saved-model.h5"
model.fit_generator(datagen.flow(train_images,train_labels,shuffle=True),epochs=50,validation_data=(test_images,test_labels),callbacks = [keras.callbacks.EarlyStopping(patience=8,verbose=1,restore_best_weights=True),keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=3,verbose=1),keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)])

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)

model.save("kanji.h5")
