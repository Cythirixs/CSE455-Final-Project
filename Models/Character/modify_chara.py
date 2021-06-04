import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split

def modify_hira():
    hira = np.load("hiragana.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

    hira = hira/np.max(hira)

    # 71 characters, 160 writers, transform image to 48*48
    train_images = np.zeros([71 * 160, 48, 48], dtype=np.float32)

    for i in range(71 * 160):
        train_images[i] = skimage.transform.resize(hira[i], (48, 48))

    #arr = np.arange(71)
    train_labels = np.repeat(0, 71*160) # create labels

    # split to train and test
    return (train_images, train_labels)

def modify_kanji():
    kanji = 879
    rows = 48
    cols = 48

    kan = np.load("kanji.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32)

    kan = kan/np.max(kan)

    train_images = np.zeros([kanji * 160, rows, cols], dtype=np.float32)

    #arr = np.arange(kanji)
    train_labels = np.repeat(1, kanji * 160)

    # 4 characters were actually hiragana, so delete these 4 extras
    for i in range( (kanji+4) * 160):
        if int(i/160) != 88 and int(i/160) != 219 and int(i/160) != 349 and int(i/160) != 457:
            if int(i/160) < 88:
                train_images[i] = skimage.transform.resize(kan[i], (rows, cols))
            if int(i/160) > 88 and int(i/160) < 219:
                train_images[i-160] = skimage.transform.resize(kan[i], (rows, cols))
            if int(i/160) > 219 and int(i/160) < 349:
                train_images[i-320] = skimage.transform.resize(kan[i], (rows, cols))
            if int(i/160) > 349 and int(i/160) < 457:
                train_images[i-480] = skimage.transform.resize(kan[i], (rows, cols))
            if int(i/160) > 457:
                train_images[i-640] = skimage.transform.resize(kan[i], (rows, cols))

    return (train_images, train_labels)

def modify_kana():
    kana = np.load("kana.npz")['arr_0'].reshape([-1, 63, 64]).astype(np.float32)

    # make the numbers range from 0 to 1
    kana = kana/np.max(kana)

    # 51 is the number of different katakana (3 are duplicates so in the end there are 48 classes)
    # 1411 writers
    # transform the size of images to 48*48
    train_images = np.zeros([51 * 1411, 48, 48], dtype=np.float32)

    for i in range(51 * 1411): # change the image size to 48*48
        train_images[i] = skimage.transform.resize(kana[i], (48, 48))

    # create labels
    #arr = np.arange(51)
    train_labels = np.repeat(2, 51*1411)

    # give the duplicates the same labels
    # for i in range(len(train_labels)):
    #     if train_labels[i] == 36:
    #         train_labels[i] = 1
    #     elif train_labels[i] == 38:
    #         train_labels[i] = 3
    #     elif train_labels[i] == 47:
    #         train_labels[i] = 2
    #     elif train_labels[i] == 37:
    #         train_labels[i] = train_labels[i] -1
    #     elif train_labels[i] >= 39 and train_labels[i] <= 46:
    #         train_labels[i] = train_labels[i] - 2
    #     elif train_labels[i] >= 48:
    #         train_labels[i] = train_labels[i] -3

    delete = [] # the 33863th and 67727th images are blank, so we delete them
    for i in range(len(train_images)):
        if (train_images[i] == np.zeros([train_images[i].shape[0],train_images[i].shape[1]],dtype=np.uint8) ).all():
            delete.append(i)

    train_images = np.delete(train_images,delete[0],axis=0)
    train_labels = np.delete(train_labels,delete[0])

    train_images = np.delete(train_images,delete[1]-1,axis=0)
    train_labels = np.delete(train_labels,delete[1]-1)

    return (train_images, train_labels)


def join():
    hira_train_image, hira_train_label = modify_hira()
    kanji_train_image, kanji_train_label = modify_kanji()
   # kana_train_image, kana_train_label = modify_kana()

    train_images = np.concatenate((hira_train_image, kanji_train_image), axis = 0)
    train_labels = np.concatenate((hira_train_label, kanji_train_label), axis = None)
        
    train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)	

    np.savez_compressed("chara_train_images.npz", train_images)
    np.savez_compressed("chara_train_labels.npz", train_labels)
    np.savez_compressed("chara_test_images.npz", test_images)
    np.savez_compressed("chara_test_labels.npz", test_labels)

join()
