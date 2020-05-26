import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = ['gplearn',
            'sklearn',
            'matplotlib',
            'numpy',
            'graphviz',
            'music21',
            'pyspark',
            'keras',
            'findspark',
            'tensorflow'
            ]

setuptools.setup(
    name="EvoComposer",  # Replace with your own username
    version="0.0.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requires)