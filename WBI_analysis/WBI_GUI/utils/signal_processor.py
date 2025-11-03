# filepath: nd2_neuron_gui/src/utils/signal_processor.py
import numpy as np
import cv2 as cv

def WBI_LUT(image):
    old_max = 80
    new_max = 255
    old_min = 20
    img = image.astype(np.float32)
    img = ((img - old_min) / (old_max - old_min)) * new_max
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    gamma = 1.3
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    img = cv.LUT(img, lut)
    
    return img