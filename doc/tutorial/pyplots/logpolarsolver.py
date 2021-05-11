import numpy as np
import imgreg.data as data
from imgreg.models.logpolar import LogPolarSolver

ref_img = np.array(data.ref_img())
mod_img = np.array(data.mod_img())

# Create the model:
lps = LogPolarSolver(ref_img, mod_img)

# The `ImageParameter`\ s of the model have `matplotlib` support via the `display` function:
lps.display([lps.REF_IMG, lps.MOD_IMG])
lps.display([lps.GAUSS_DIFF_REF_IMG, lps.GAUSS_DIFF_MOD_IMG])
lps.display([lps.FOURIER_REF_IMG, lps.FOURIER_MOD_IMG])
lps.display([lps.WARPED_FOURIER_MOD_IMG, lps.WARPED_FOURIER_REF_IMG])
lps.display([lps.RECOVERED_ROT_SCALE_IMG, lps.REF_IMG])
lps.display([lps.RECOVERED_ROT_SCALE_TR_IMG, lps.REF_IMG])
