import numpy as np
from skimage import transform

class ResizeGenerator:
    def __init__(self, batch_size=32, factor=4.0, preprocess=None):
        self.batch_size = batch_size
        self.factor = factor
        self.preprocess = preprocess

    def generate_ps(self, inp, N):
        if self.preprocess is not None:
            transp, mean, std = self.preprocess
            # inp = inp.transpose(*transp)
            inp = (inp - mean) / std

        ps = []
        for _ in range(N):
            shape = inp.shape
            assert len(shape)==3 and shape[2] == 3
            p_small = np.random.randn(int(shape[0]/self.factor), int(shape[1]/self.factor), shape[2])
            #if (_ == 0):
            #    print (p_small.shape)
            p = transform.resize(p_small, inp.shape)
            ps.append(p)
        ps = np.stack(ps, axis=0)

        if self.preprocess is not None:
            rev_transp = np.argsort(transp)
            ps = ps * std
            ps = ps.transpose(0, *(rev_transp+1))

        return ps
