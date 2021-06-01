import numpy as np
import matplotlib.pyplot as plt
import imgreg.data as data
from imgreg.util.methods import ImageMethods

img = np.array(data.mod_img())

# Compute the sinogram with the exponential weighting filter and project the image values
# to the axis corresponding to the theta angles.
sinogram_project = ImageMethods.sinogram_project(img, exp_filter_val=1000)
plt.plot(sinogram_project)
plt.xlabel("theta [°]")
plt.show()
