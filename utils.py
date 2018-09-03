import PIL
import numpy as np
from matplotlib import pyplot as plt
import skimage, skimage.transform

# Image resize
def imresize(img, height=None, width=None):
    # load image
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]

    return skimage.transform.resize(img, (int(ny), int(nx)), mode='constant')

# Heat map visualization
def show_heatmaps(imgs, masks, K, enhance=1, title=None, cmap='gist_rainbow'):

    if K > 0:
        _cmap = plt.cm.get_cmap(cmap)
        colors = [np.array(_cmap(i)[:3]) for i in np.arange(0,1,1/K)]
    plt.figure(figsize=(4 * len(imgs), 4))
    if title is not None:
        plt.suptitle(title+'\n', fontsize=24).set_y(1.05)
    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i + 1)

        img = imgs[i]
        if img.max()<=1:
            img *= 255
        img = np.array(PIL.ImageEnhance.Color(PIL.Image.fromarray(np.uint8(img))).enhance(enhance))
        plt.imshow(img)
        plt.axis('off')
        for k in range(K):
            layer = np.ones((*img.shape[:2],4))
            for c in range(3): layer[:,:,c] *= colors[k][c]
            mask = masks[i][k]
            layer[:,:,3] = mask
            plt.imshow(layer)
            plt.axis('off')

    plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.show()


