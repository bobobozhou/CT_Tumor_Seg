import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from utilizes import *

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

img = np.load('input_var.npy')[0,:,:]

pos = np.stack(np.mgrid[0:img.shape[1], 0:img.shape[0]], axis=2)
rv = multivariate_normal([img.shape[1] // 2, img.shape[0] // 2 - 15], (img.shape[1] // 9) * (img.shape[0] // 9))
pred = rv.pdf(pos)
pred = (pred - pred.min()) / (pred.max() - pred.min())
# pred = 0.5 + 0.2 * (pred - 0.5)

map, kl = crf_segment(img, pred, iter=88, n_labels=2)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1); plt.imshow(img);
plt.subplot(1,2,2); plt.imshow(map);
plt.show()


