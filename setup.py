import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="c45-decision-tree",
    version="1.0.1",
    author="Novandi Kevin Pratama",
    author_email="kevinpret27@gmail.com",
    description="Library for C4.5 Decision Tree Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/novandikp/DecisionTreeC45",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)