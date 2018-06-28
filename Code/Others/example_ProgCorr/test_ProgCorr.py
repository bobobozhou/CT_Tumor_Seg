import math
import matplotlib.pyplot as plt
import numpy as np

from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate
from PIL import Image, ImageDraw
from scipy.ndimage.morphology import binary_dilation

# create simulated images
image = np.zeros((600, 600))
rr, cc = ellipse(300, 350, 50, 110); image[rr, cc] = 1
rr, cc = ellipse(400, 450, 50, 110); image[rr, cc] = 1
rr, cc = ellipse(100, 450, 50, 110); image[rr, cc] = 1
image = rotate(image, angle=15, order=0)

# detect regions and labels
label_img = label(image)
regions = regionprops(label_img)

# create connection mask
if len(regions) > 1:
    img_conn = Image.new('L', (image.shape[0], image.shape[1]), 0)
    polygon_pts = []
    area_min = float("inf")
    for props in regions:
        y0, x0 = props.centroid
        polygon_pts.append((x0, y0))

        # store the structure element for dilation
        if props.area < area_min:
            se = props.convex_image

    ImageDraw.Draw(img_conn).polygon(polygon_pts, outline=1, fill=1)
    img_conn = np.array(img_conn)
    img_conn_dil = binary_dilation(img_conn, structure=se)

    img_fin = np.logical_or(img_conn_dil, image)

    # visualize
    plt.subplot(131)
    plt.imshow(img_conn, cmap=plt.cm.gray)
    plt.subplot(132)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.subplot(133)
    plt.imshow(img_fin, cmap=plt.cm.gray)
    plt.show()