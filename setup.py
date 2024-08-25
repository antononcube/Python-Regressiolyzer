import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Regressionizer",
    version="0.1.0",
    author="Anton Antonov",
    author_email="antononcube@posteo.net",
    description="Regression workflows package based on Linear Regression and Quantile Regression.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antononcube/Python-packages/tree/main/LatentSemanticAnalyzer",
    install_requires=['numpy', 'scipy', 'pandas', 'plotly', 'datetime'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords=['regression','quantile', 'quantile regression', 'linear', 'linear regression', 'workflow'],
    package_data={'': ['resources/*.csv', 'resources/*.csv.gz']},
    python_requires='>=3.7',
)