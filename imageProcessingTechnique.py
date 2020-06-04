import numpy as np
import os
import cv2
import random
import regionGrowing
import splitImageIntoRegions
import matplotlib.pyplot as plt

def read_save_image(image):

    cv2.imshow('image', image)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('messigray.png',image)
        cv2.destroyAllWindows()

def noisy(noise_typ,image):
   if noise_typ == "gauss":
       gaussian_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
       cv2.randn(gaussian_noise, 128, 20)

       gaussian_noise = (gaussian_noise * 0.5).astype(np.uint8)
       noisy_image = cv2.add(image, gaussian_noise)
       read_save_image(noisy_image)

   elif noise_typ == "s&p":
       uniform_noise = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
       cv2.randu(uniform_noise, 0, 255)
       impulse_noise = uniform_noise.copy()

       ret, impulse_noise = cv2.threshold(impulse_noise, 250, 255, cv2.THRESH_BINARY)
       noisy_image = cv2.add(image, impulse_noise)
       read_save_image(noisy_image)


def deNoise(denois_type, image):

    if denois_type == "mean":
        processed_image = cv2.blur(image, (5, 5))

        read_save_image(processed_image)

    elif denois_type == "median":
        median = cv2.medianBlur(image, 5)

        read_save_image(median)

def edgeDetection(edgDetType, image):
    if edgDetType == "prewitt":
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt = kernelx + kernely
        prewittImage = cv2.filter2D(image, -1, prewitt)
        read_save_image(prewittImage)

    elif edgDetType == "sobel":
        image_X = cv2.Sobel(image, cv2.CV_8UC1, 1, 0)
        image_Y = cv2.Sobel(image, cv2.CV_8UC1, 0, 1)
        sobel = cv2.add(image_X, image_Y)
        sobelImage = np.uint8(sobel)
        read_save_image(sobelImage)

def imageSegmentation(segmType, image):
    if segmType == "adaptiveSegmentation":
        th1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        read_save_image(th1)
    elif segmType == "regionGrowing":
        exemple = regionGrowing.regionGrow('messi5.jpg', 10)
        exemple.ApplyRegionGrow()
    elif segmType == "regionSplitting":
         img = cv2.imread('messi5.jpg', 1)
         quadtree = splitImageIntoRegions.QuadTree().insert(img)
         plt.imshow(quadtree.get_image(6))
         plt.show()


    elif segmType == "regionSplitting_merging":
        img = cv2.imread('messi5.jpg', 1)
        quadtree = splitImageIntoRegions.QuadTree().insert(img)
        plt.imshow(quadtree.get_image(7))
        plt.show()



