import numpy as np
import matplotlib.pyplot as plt
import imgreg.data as data
from imgreg.util.methods import ImageMethods

img = np.array(data.mod_img())

# Compute the sinogram using the radon transform
sinogram = ImageMethods.sinogram(img)
plt.imshow(sinogram, aspect=0.1)
plt.show()

# Compute the same sinogram but apply an exponential weighting filter
sinogram = ImageMethods.sinogram(img, exp_filter_val=1000)
plt.imshow(sinogram, aspect=0.1)
plt.show()
