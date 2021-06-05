import imageio
import skimage.transform
import numpy as np
from matplotlib import pyplot as plt

from preprocess_input import process
import cv2

def create_histogram(filepath):
    image = imageio.imread(filepath, as_gray = True)
    w, h = image.shape
    #print(w, h)
    x_hist = np.zeros(h)
    y_hist = np.zeros(w)

    for i in range(0, w):
        for j in range(0, h):
            if image[i][j] == 255.0:
                x_hist[j] = x_hist[j] + 1
                y_hist[i] = y_hist[i] + 1

    # x = np.arange(0, len(x_hist))
    # plt.bar(x, x_hist, width = 3, color='black')
    # plt.savefig('x_hist.png')
    # plt.show()
    
    return (x_hist, y_hist, image)

def create_histogram_from_array(array):
    w = len(array)
    h = len(array[0])

    x_hist = np.zeros(h)
    y_hist = np.zeros(w)

    for i in range(0, w):
        for j in range(0, h):
            if array[i][j] == 255.0:
                x_hist[j] = x_hist[j] + 1
                y_hist[i] = y_hist[i] + 1
    

    return (x_hist, y_hist)

def trim_img(x_hist, y_hist, image, thresh, space):
    y_start = 0
    y_end = 0
    for i in range(0, len(y_hist)):
        if y_hist[i] > thresh and y_start == 0:
            y_start = i
        if y_hist[-i] > thresh and y_end == 0:
            y_end = len(y_hist)-i-1
        if y_start != 0 and y_end != 0:
            break

    x_start = 0
    x_end = 0
    for i in range(0, len(x_hist)):
        if x_hist[i] > thresh and x_start == 0:
            x_start = i
        if x_hist[-i] > thresh and x_end == 0:
            x_end = len(x_hist)-i-1
        if x_start != 0 and x_end != 0:
            break
    x_start = 0 if x_start - space < 0 else x_start - space
    y_start = 0 if y_start - space < 0 else y_start - space
    x_end = len(x_hist)-1 if x_end + space > len(x_hist)-1 else x_end + space
    y_end = len(y_hist)-1 if y_end + space > len(y_hist)-1 else y_end + space

    print(x_start, x_end, y_start, y_end)

    crop = image[y_start:y_end, x_start:x_end]
    y_hist_crop = y_hist[y_start:y_end]
    x_hist_crop = x_hist[x_start:x_end]

    imageio.imwrite("crop_img.jpg", crop)
    return (x_hist_crop, y_hist_crop, crop)
    
def find_rect(x_hist, y_hist, image, filter):
    space = 10
    expected = 120
    # x = np.arange(0, len(y_hist))
    # plt.bar(x, y_hist, width = 0.1)
    # plt.show()

    x_segment = []
    start = 0
    x_sum = 0
    for i in range(0, len(x_hist)):
        if x_hist[i] != 0 and start == 0:
            start = i
        elif x_hist[i] == 0 and start != 0:
            if abs(i - start) > expected:
                #x_segment.append((int)((start + i)/2))
                x_segment.append(start - space)
                x_segment.append(i + space)
                start = 0
                x_sum = 0
            elif x_sum < filter:
                start = 0
                x_sum = 0
        if start != 0:
            x_sum = x_sum + x_hist[i]
        
        
    #x_segment.append(len(x_hist))
    print(x_segment)

    y_segment = []
    start = 0
    y_sum = 0
    for i in range(0, len(y_hist)):
        if y_hist[i] != 0 and start == 0:
            start = i
        elif y_hist[i] == 0 and start != 0:
            #print(start, i, y_sum)
            if abs(i - start) > expected:
                #y_segment.append((int)((start + i)/2))
                y_segment.append(start - space)
                y_segment.append(i + space)
                start = 0
                y_sum = 0
            elif y_sum < filter:
                start = 0
                y_sum = 0
        if start != 0:
            y_sum = y_sum + y_hist[i]

    #y_segment.append(len(y_hist))
    print(y_segment)

    show = cv2.imread("crop_img.jpg")

    for x in x_segment:
        cv2.line(show, (x, 0), (x, len(y_hist)), (0, 0, 255), 2)   

    color = [(0, 0, 255), (0, 0, 255)]
    i = 0
    for y in y_segment:
        cv2.line(show, (0, y), (len(x_hist), y), color[i%2], 2)
        i = i + 1

    cv2.imshow("Image", show)
    cv2.imwrite("bounding_boxes.jpg", show)
    cv2.waitKey(0)

    return (x_segment, y_segment)

def create_rect(x_hist, y_hist, image):
    (x_segment, y_segment) = find_rect(x_hist, y_hist, image, 20)

    n = 0
    for i in range(0, (len(x_segment)//2)):
        x1 = x_segment[i*2]
        x2 = x_segment[i*2+1]
        for j in range(0, (len(y_segment)//2)):
            y1 = y_segment[j*2]
            y2 = y_segment[j*2+1]
            print(x1, x2, y1, y2)
            #characters.append(image[prev_x:x, prev_y:y])
            path = "./characters/chara_{0}.jpg".format(n)

            crop = image[y1:y2, x1:x2]

            imageio.imwrite(path, crop)

            x_temp, y_temp = create_histogram_from_array(crop)
            x_trim, y_trim = find_rect(x_temp, y_temp, crop, 10)
            
            xt1 = 0
            xt2 = 0 
            yt1 = 0 
            yt2 = 0
            if len(x_trim) == 2:
                xt1 = x_trim[0]
                xt2 = x_trim[-1]
            else:
                xt2 = x2-x1-1
            
            if len(y_trim) == 2:
                yt1 = y_trim[0]
                yt2 = y_trim[-1]
            else:
                yt2 = y2-y1-1
            
            final = crop[yt1:yt2, xt1:xt2]
            path2 = "./characters/chara_trim_{0}.jpg".format(n)
            imageio.imwrite(path2, final)

            inc = 30
            size = max(yt2-yt1+inc, xt2-xt1+inc)
            square = np.zeros([size, size], np.uint8)
            ax,ay = (size - (xt2-xt1))//2,(size - (yt2-yt1))//2
            print(ax, ay)
           
            square[ay:(yt2-yt1)+ay,ax:ax+(xt2-xt1)] = final

            path3 = "./characters/chara_square_{0}.jpg".format(n)
            imageio.imwrite(path3, square)
            n = n + 1

    # print(len(characters))
    # for i in (0, len(characters)-1):
    #     path = "./characters/chara_{0}.jpg".format(i)
    #     imageio.imwrite(path, characters[i])



def main():
    x_hist, y_hist, image = create_histogram("thresh_img.jpg")
    x_hist, y_hist, image = trim_img(x_hist, y_hist, image, 5, 50)
    create_rect(x_hist, y_hist, image)

#process("../test_model/hello1.jpg")
main()