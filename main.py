import numpy as np
import cv2
import imageProcessingTechnique

img = cv2.imread('messi5.jpg', 0)

grascale_image = imageProcessingTechnique.read_save_image(img)
gaussian_noise_image = imageProcessingTechnique.noisy("gauss", img)
salt_pepper_noise_image = imageProcessingTechnique.noisy("s&p", img)
mean_deNoise_image = imageProcessingTechnique.deNoise("mean", img)
median_deNoise_image = imageProcessingTechnique.deNoise("median", img)
prewitt_edgDet_image = imageProcessingTechnique.edgeDetection("prewitt", img)
sobel_edgDet_image = imageProcessingTechnique.edgeDetection("sobel", img)
adaptThr_imSeg_image = imageProcessingTechnique.imageSegmentation("adaptiveSegmentation", img)
regGr_imSeg_image = imageProcessingTechnique.imageSegmentation("regionGrowing", img)
regSpl_imSeg_image = imageProcessingTechnique.imageSegmentation("regionSplitting", img)
regSplMer_imSeg_image = imageProcessingTechnique.imageSegmentation("regionSplittingMerging", img)







