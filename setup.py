from distutils.core import setup

setup(
    name="pydct",
    version="1.0.0",
    author="Jonas Haag",
    author_email="jonas@lophus.org",
    url="https://github.com/jonashaag/pydct",
    license="2-clause BSD",
    description="Discrete Cosine Transform (DCT) for Python using SciPy or Tensorflow",
    py_modules=["pydct", "pydct.scipy", "pydct.tf"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    extras_require={
        "test": [
            "pytest",
            "librosa",
            "scipy",
            "numpy",
            "tensorflow",
            "black",
            "isort",
            "flake8",
        ]
    },
)
