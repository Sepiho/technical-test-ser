#!/usr/bin/env python

import setuptools

setuptools.setup(
    name="servier",
    version="0.0.1",
    author="Sophie Sebille",
    author_email="sophie.sebille@laposte.net",
    description="Technical test",
    url="https://github.com/Sepiho/technical-test-ser",
    packages=setuptools.find_packages(),
    install_requires=["pandas", "sklearn", "flask", "keras", "tensorflow", "numpy==1.19.2"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'split=servier.read_split_data:main',
            'train=servier.train:main',
            'evaluate=servier.evaluate:main',
            'predict=servier.predict:main',
        ]
    }
)