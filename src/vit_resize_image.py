import cv2
import numpy as np

def ViT_resize_image(images, height, width):
    preprocessed_images = []
    for image in images:
        resized = cv2.resize(image, (height,width), interpolation = cv2.INTER_AREA)
        resized_image =  np.array(resized) / 255.0
        resized_image = resized_image.astype(np.float32)
        preprocessed_images.append(resized_image)
    preprocessed_images = np.array(preprocessed_images)
    return preprocessed_images