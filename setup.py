#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="fomo25-inference",
    version="1.0.0",
    description="FOMO25 Challenge Inference Pipeline",
    author="FOMO25 Challenge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0,<2.3.0",
        "lightning>=2.1.4",
        "nibabel>=5.0.0",
        "numpy>=1.23,<2.0",
        "yucca==2.2.6",
        "opencv-python>=4.8.0",
        "einops>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "predict-task1=inference.predict_task1:main",
            "predict-task2=inference.predict_task2:main",
            "predict-task3=inference.predict_task3:main",
        ],
    },
)
