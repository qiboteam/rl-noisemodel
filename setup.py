import os
import re
from setuptools import find_packages, setup

PACKAGE = "rlnoise"

# Returns the qibo version
def get_version():
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE).readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)

setup(
    name="rlnoise",
    version=get_version(),
    description="Quantum noise modelling through reinforcement learning.",
    author="Simone Bordoni, Piergiorgio Buttarini, Andrea Papaluca, Alejandro Sopena",
    author_email="",
    url="https://github.com/qiboteam/rl-noisemodel",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ["*.out", "*.yml"]},
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "matplotlib",
        "numpy",
        "gymnasium",
        "qibo==0.2.7",
        "qibojit",
        "stable_baselines3==2.0.0",
        "tqdm",
        "rich",
        "gym",
        "shimmy==0.2.1"
       
    ],
    extras_require={
        'hardware':  ["qiboconnection @ git+ssh://git@github.com/AlejandroSopena/qiboconnection.git@qibo_0.2.8"],
    },
    python_requires=">3.10.10"
)