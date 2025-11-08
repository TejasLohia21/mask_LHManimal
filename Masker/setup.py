from setuptools import setup, find_packages
setup(
    name="masker",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.26,<2.3",
        "opencv-python>=4.7,<4.13",
    ],
)
