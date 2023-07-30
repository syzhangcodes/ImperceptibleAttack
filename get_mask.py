import numpy as np
import pywt
import pywt.data
import cv2

def hfp_dft(img_c):
    #DFT
    f = np.fft.fft2(img_c)
    fshift = np.fft.fftshift(f)

    #High pass
    rows, cols = img_c.shape
    crow,ccol = int(rows/2), int(cols/2)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0

    #IDFT
    ishift = np.fft.ifftshift(fshift)
    iimg = np.fft.ifft2(ishift)
    iimg = np.abs(iimg)

    iimg = iimg - np.min(iimg)
    iimg = iimg / np.max(iimg)

    return iimg


def dct2(img_c):
    img_c_dct = cv2.dct(img_c)
    #High pass
    rows, cols = img_c.shape
    # crow,ccol = int(rows/2), int(cols/2)
    img_c_dct[0:30, 0:30] = 0.0
    # img_c_dct[0:rows, 120:cols] = 0.0
    return img_c_dct

def idct2(img_c_dct):
    img_c_recor = cv2.idct(img_c_dct)
    return img_c_recor

def color_img_dct(img):
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]

    img_dct = np.expand_dims(dct2(img_r), axis=2)
    img_dct = np.append(img_dct, np.expand_dims(dct2(img_g), axis=2), axis=2)
    img_dct = np.append(img_dct, np.expand_dims(dct2(img_b), axis=2), axis=2)

    return img_dct

def color_img_idct(img_dct):
    img_r_dct = img_dct[:,:,0]
    img_g_dct = img_dct[:,:,1]
    img_b_dct = img_dct[:,:,2]

    img_recor = np.expand_dims(idct2(img_r_dct), axis=2)
    img_recor = np.append(img_recor, np.expand_dims(idct2(img_g_dct), axis=2), axis=2)
    img_recor = np.append(img_recor, np.expand_dims(idct2(img_b_dct), axis=2), axis=2)

    return img_recor


def dwt_hpf(img):
    coeffs2 = pywt.dwt2(img, 'sym2')
    cA, (cH, cV, cD) = coeffs2
    cA = np.where(cA != 0, 0.0, 0.0)
    img_recoer = pywt.idwt2((cA, (cH, cV, cD)), 'sym2')
    return img_recoer

def dwt_hpf(img):
    img_re = np.zeros(img.shape)
    for i in range(img.shape[2]):
        channel = img[:,:,i]        
        coeffs = pywt.wavedecn(channel, 'sym2', level=1)
        coeffs[0][:,:] = 0
        channel_re = pywt.waverecn(coeffs, 'sym2')        
        img_re[:,:,i] = channel_re
    return img_re

def get_mask(img, gama, method='dft'):

    if method == 'dft':
        iimg = np.expand_dims(hfp_dft(img[:,:,0]), axis=2)
        iimg = np.append(iimg, np.expand_dims(hfp_dft(img[:,:,1]), axis=2), axis=2)
        iimg = np.append(iimg, np.expand_dims(hfp_dft(img[:,:,2]), axis=2), axis=2)
    elif method == 'dct':
        img_dct = color_img_dct(img)
        iimg = color_img_idct(img_dct)
    elif method == 'dwt':
        iimg = dwt_hpf(img)

    threshold = np.sort(iimg.flatten())[int((iimg.flatten().shape[0]-1) * (1 - gama))]

    iimg = np.where(iimg >= threshold, 1.0, 0.0)

    return iimg