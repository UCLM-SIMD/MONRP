from setuptools import setup

setup(name='monrp',
      version='1.0',
      author="Víctor Pérez-Piqueras",
      author_email="victor.perezpiqueras@uclm.es",
      description="Multi-Ojective Next Release Problem set of algorithms",
      packages=["algorithms", "datasets",
                "models", "evaluation"],
      install_requires=[
          'imageio',
          "pandas",
          "numpy",
          "scipy",
          "scikit-learn",
          "matplotlib"
      ],
      python_requires=">=3.8",
      )
