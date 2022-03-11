import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(name='monrp',
      version='1.0',
      author="Víctor Pérez-Piqueras",
      author_email="victor.perezpiqueras@uclm.es",
      description="Multi-Ojective Next Release Problem set of algorithms",
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/UCLM-SIMD/MONRP",
      license="MIT",
      classifiers=[
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.8"
      ],
      packages=find_packages(exclude=("test_*",)),
      install_requires=[
          'imageio',
          "pandas",
          "numpy",
          "scipy",
          "scikit-learn",
          "matplotlib"
      ],
      python_requires=">=3.8",
      include_package_data=True,
      )
