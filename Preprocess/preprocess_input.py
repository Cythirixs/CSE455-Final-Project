import imageio
import skimage.transform
import skimage.filters
import numpy as np
import cv2


def to_grayscale(filepath):
    image = imageio.imread(filepath, as_gray = True)
    #image = skimage.transform.resize(image, (48, 48))
    imageio.imwrite("gray_img.jpg", image)
    return image

def invert(image):
    invert = lambda x: 255.0-x
    func = np.vectorize(invert)
    image = func(image)
    imageio.imwrite("invert_img.jpg", image)
    return image

def threshold(image, thresh):
    threshing = lambda x: 0.0 if x < thresh else 255.0
    func = np.vectorize(threshing)
    threshed = func(image) 
    #ret,thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    imageio.imwrite("thresh_img.jpg", threshed)
    return threshed
    
def zoom_in(image):
    print("hay")

def blur(image, sigma):
    blur = skimage.filters.gaussian(image, sigma=(sigma, sigma), truncate=3.5, multichannel=False)
    imageio.imwrite("blur_img.jpg", blur)
    return blur

def process(filename):
    image = to_grayscale(filename)
    image = invert(image)
    image = threshold(image, 100)
    #image = blur(image, 10)
    
