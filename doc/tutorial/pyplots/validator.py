import numpy as np
import imgreg.data as data
from imgreg.models.validator import Validator
from imgreg.util.methods import ImageMethods

ref_img = np.array(data.ref_img())

# modify the image using an affine transformation
img = ImageMethods.compute_rts(ref_img, angle=2, translation=(6, 2))

# Create the model:
val = Validator(img, ref_img)

# The `ImageParameter`\ s of the model have `matplotlib` support via the `display` function:
val.display([val.ABSOLUTE_DIFFERENCE_IMG, val.SQUARED_DIFFERENCE_IMG])

# Increase the overlap to the reference image
val.IMG.value = ImageMethods.compute_rts(ref_img, angle=1, translation=(1, 2))

# Note how the difference images show less pronounced differences with increased overlap
val.display([val.ABSOLUTE_DIFFERENCE_IMG, val.SQUARED_DIFFERENCE_IMG])
