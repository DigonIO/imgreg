from setuptools import setup

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="imgreg",
    version="1.0.1",
    description="Image registration models with extendable interfaces.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Fabian A. Preiss, Jendrik A. Potyka",
    author_email="devops@digon.io",
    license="GPLv3",
    packages=[
        "imgreg",
        "imgreg.data",
        "imgreg.models",
        "imgreg.util",
        "imgreg.models.logpolar",
        "imgreg.models.angleselect",
        "imgreg.models.radon",
        "imgreg.models.validator",
    ],
    keywords="image reg",
    package_data={"imgreg": ["data/ref_img.jpg", "data/mod_img.jpg"]},
    include_package_data=True,
    install_requires=[
        "numpy==1.20.2",
        "scipy==1.6.3",
        "scikit-image==0.18.1",
        "matplotlib==3.4.1",
        "typeguard==2.12.0",
        "graphviz==0.16",
        "Pillow==8.2.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries",
        "Operating System :: Unix",
        "Operating System :: POSIX",
    ],
    zip_safe=False,
)
