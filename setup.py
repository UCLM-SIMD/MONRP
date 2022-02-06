from setuptools import setup

setup(name='monrp',
      version='1.0',
      author="Víctor Pérez-Piqueras",
      author_email="victor.perezpiqueras@uclm.es",
      description="Multi-Ojective Next Release Problem set of algorithms",
      package_dir={"algorithms": "algorithms", "datasets": "datasets",
                   "models": "models", "evaluation": "evaluation"},
      packages=["algorithms", "datasets",
                "models", "evaluation"],
      python_requires=">=3.8",
      )
