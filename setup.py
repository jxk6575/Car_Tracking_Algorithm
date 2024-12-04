from setuptools import setup, find_packages

setup(
    name="car_tracking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.7.0",
        "filterpy>=1.4.5",
        "opencv-python>=4.5.0",
        "pyyaml>=5.4.0",
        "pytest>=6.0.0",
    ],
) 