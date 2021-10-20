import unittest

# import your test modules
from algorithms.EDA.bivariate.MIMIC.test_mimic_algorithm import MIMICTestCase
from algorithms.EDA.UMDA.test_umda_algorithm import UMDATestCase
from algorithms.GRASP.test_grasp_algorithm import GRASPTestCase




# initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(MIMICTestCase))
suite.addTests(loader.loadTestsFromModule(UMDATestCase))
suite.addTests(loader.loadTestsFromModule(GRASPTestCase))




# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=5)
result = runner.run(suite)


# run: python -m unittest unit_tests.py
