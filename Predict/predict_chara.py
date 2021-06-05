from tensorflow import keras
import imageio
import skimage.transform
import numpy as np
import subprocess

from kanjijapanese import kanji_label
from hiraganajapanese import hira_label

def scale_input(filepath, size):
    image = imageio.imread(filepath, as_gray = True)
    image = skimage.transform.resize(image, (size, size))
    imageio.imwrite("image_gray.png", image)
    image = np.reshape(image, [1, size, size, 1])
    #print(image)
    return image

def load():
    size = 48
    #img = scale_input("../Preprocess/thresh_img.jpg")
    #img = scale_input("../test_model/kanji_test4.jpg")
    # img3 = scale_input("../test_model/kanji_test2.png")
    img = scale_input("../Preprocess/characters/chara_square_0.jpg", size)
    img1 = scale_input("../Preprocess/characters/chara_square_1.jpg", size)
    img2 = scale_input("../Preprocess/characters/chara_square_2.jpg", size)
    img3 = scale_input("../Preprocess/characters/chara_square_3.jpg", size)
    img4 = scale_input("../Preprocess/characters/chara_square_4.jpg", size)
    
    to_predict = np.concatenate((img, img1, img2, img3, img4), axis=0)
    #to_predict = np.concatenate((img, img1), axis=0)
    print(to_predict.shape)
    return to_predict

def predictImage(to_predict):
    size = 48
    chara_model = keras.models.load_model('./models/character48.h5')
    hira_model = keras.models.load_model('./models/hiragana48.h5')
    kanji_model = keras.models.load_model('./models/kanji.h5')

    chara_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    hira_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    kanji_model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    #chara_type = chara_model.predict_classes(to_predict)
    chara_type = [0, 0, 0, 0, 0]
    print(chara_type)

    result = ""

    for i in range(0, len(chara_type)):
        temp = np.reshape(to_predict[i], [1, size, size, 1])
        if(chara_type[i] == 0):
            hira_type = hira_model.predict_classes(temp)
            result = result + hira_label[hira_type[0]]
            print("hira:", hira_type, hira_label[hira_type[0]])
        else:
            kanji_type = kanji_model.predict_classes(temp)
            result = result + kanji_label[kanji_type[0]]
            print("kanji:", kanji_type, kanji_label[kanji_type[0]])

    return result

def getIchiranInfo(res):
    print(res)
    process = "C:/Users/alexm/quicklisp/local-projects/ichiran/ichiran-cli.exe"
    subprocess.run([process, "-i", res])

def main():
    to_predict = load()
    result = predictImage(to_predict)
    getIchiranInfo(result)
    
main()



