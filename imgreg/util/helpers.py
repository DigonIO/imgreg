"""
Collection of useful helper functions in specific usecases

Author: Fabian A. Preiss
"""
from typing import Generator, Iterable, Sequence

import numpy as np
from PIL import Image

from imgreg.util.io import DirectoryView
from imgreg.util.methods import ImageMethods
from imgreg.util.params import ParameterError
from imgreg.util.solver import Solver


def image_file_gen(
    dview: DirectoryView, step: int = 1
) -> Generator[np.ndarray, None, None]:
    """
    Generate the images from a given DirectoryView as numpy arrays

    Parameters
    ----------
    dview : DirectoryView
        Generate images from top level folder
    step : int
        Stepsize in which files are taken
    """
    for file_path in dview.file_path_generator(step):
        yield 1.0 * np.array(Image.open(file_path))


def solver_gen(
    dview: DirectoryView, solver: Solver, step: int = 1
) -> Generator[Solver, None, None]:
    """Sweep the MOD_IMG parameters of a solver from a DirectoryView with a given stepsize.

    Parameters
    ----------
    dview : DirectoryView
        Generate images from top level folder
    solver : Solver
        Instance of a solver that
    step : int
        Stepsize in which files are taken
    """
    image_file_g = image_file_gen(dview, step)
    try:
        solver.REF_IMG.value
    except ParameterError as err:
        raise ParameterError(
            "Solver is expected to have the REF_IMG parameter set."
        ) from err
    try:
        solver.MOD_IMG.value = next(image_file_g)
    except StopIteration:
        pass
    yield solver
    for arr in image_file_g:
        solver.MOD_IMG.value = arr
        yield solver


def rot_tr_gen(solvers: Iterable[Solver]) -> Generator[np.ndarray, None, None]:
    """
    Sweep over solver instances and return `tr_x` `tr_y` `tr_err` `rot` `rot_err` `NormRel_L2`

    Parameters
    ----------
    solvers : Iterable
        An iterable of solver instances
    """
    for solv in solvers:
        yield np.array(
            (
                *solv.RECOVERED_TRANSLATION.value,
                *solv.RECOVERED_ROTATION.value,
                ImageMethods.norm_rel_l2(
                    solv.RECOVERED_ROT_TR_IMG.value, solv.REF_IMG.value
                ),
            )
        )


def rot_scale_tr_gen(solvers: Iterable[Solver]) -> Generator[np.ndarray, None, None]:
    """
    Sweep over solver instances and return the rotations, scales and translations

    Parameters
    ----------
    solvers : Iterable
        An iterable of solver instances

    Returns
    -------
    np.array
        Containing `tr_x` `tr_y` `tr_err` `rot` `rot_err` `scale` `scale_err` `NormRel_L2`
    """
    for solv in solvers:
        yield np.array(
            (
                *solv.RECOVERED_TRANSLATION.value,
                *solv.RECOVERED_ROTATION.value,
                *solv.RECOVERED_SCALE.value,
                ImageMethods.norm_rel_l2(
                    solv.RECOVERED_ROT_SCALE_TR_IMG.value, solv.REF_IMG.value
                ),
            )
        )


def image_save_back_tf(
    rot_tr_arr: np.ndarray, fnames: Sequence[str], src_path: str, dest_path: str
):
    """
    Creates backtransformed images

    For an external program that compares images `*_A.*` with `*_B.*` this prepares images
    such that::

      test00001.jpg -> test00001_A.png
      test00021.jpg -> TR -> test00001_B.png, test00021_A.png

      test00221.jpg -> TR -> test00201_B.png, test00221_A.png
      test00241.jpg -> TR -> test00221_B.png

    where TR denotes the backtransformation.
    """
    for i, fname in enumerate(fnames):
        if i == 0:
            pass
        src_arr = np.array(Image.open(src_path + "/" + fname))
        dest_arr = ImageMethods.compute_rts(
            src_arr,
            angle=rot_tr_arr[i][3],
            translation=rot_tr_arr[i][:2],
            inverse=True,
        )
        dest_img = Image.fromarray(dest_arr).convert("L")
        if i != len(fnames) - 1:
            out_a = dest_path + "/" + fname.split(".")[0] + "_A.png"
            dest_img.save(out_a)
        if i != 0:
            out_b = dest_path + "/" + fnames[i - 1].split(".")[0] + "_B.png"
            dest_img.save(out_b)
