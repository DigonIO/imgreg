import numpy as np
import imgreg.data as data
from imgreg.models.radon import RadonSolver

ref_img = np.array(data.ref_img())
mod_img = np.array(data.mod_img())

# Create the model:
ras = RadonSolver(ref_img, mod_img)

# The ImageParameters of the model have matplotlib support via the display function:
ras.display([ras.REF_IMG, ras.MOD_IMG])
ras.display([ras.RECOVERED_ROT_IMG, ras.REF_IMG])
ras.display([ras.RECOVERED_ROT_TR_IMG, ras.REF_IMG])
