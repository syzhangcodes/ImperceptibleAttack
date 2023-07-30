import numpy as np
from scipy.fftpack import fft2, ifft2
from keras.applications.resnet50 import decode_predictions


def phase_search(ori_img, tgt_img, model, label=None):
    ori_label = np.argmax(model.predict(np.expand_dims(ori_img, axis=0)))
    tgt_label = np.argmax(model.predict(np.expand_dims(tgt_img, axis=0)))
    if tgt_label != label:
        raise ValueError("Target image is not the target label")

    ori_fft = fft2(ori_img)
    ori_pha = np.angle(ori_fft)
    tgt_fft = fft2(tgt_img)
    tgt_mag, tgt_pha = np.abs(tgt_fft), np.angle(tgt_fft)

    
    tmp_fft = tgt_mag * np.exp(1j * ori_pha)
    tmp_img = np.abs(ifft2(tmp_fft))

    if np.argmax(model.predict(np.expand_dims(tmp_img, axis=0))) == tgt_label:
        # * No need to search
        return tgt_img, 1

    i=0
    left = 0.0
    right = 1.0
    while (i < 8):
        mid = (left + right) / 2
        print("mid: ", mid)
        mid_pha = tgt_pha * (1 - mid) + ori_pha * mid
        tmp_fft = tgt_mag * np.exp(1j * mid_pha)
        tmp_img = np.abs(ifft2(tmp_fft))
        tmp_label = np.argmax(model.predict(np.expand_dims(tmp_img, axis=0)))
        print("tmp_label: ", decode_predictions(model.predict(np.expand_dims(tmp_img, axis=0)), top=1)[0])
        if tmp_label == tgt_label:
            left = mid
        else:
            right = mid
        i += 1
    
    r_pha = tgt_pha * (1 - left) + ori_pha * left
    r_fft = tgt_mag * np.exp(1j * r_pha)
    r_img = np.abs(ifft2(r_fft))
    if tmp_label != tgt_label:
        # * Search failed
        flag = 0
        return r_img, flag
    else:
        # * Success
        flag = 1
        return r_img, flag

    
