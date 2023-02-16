import json
import unittest
from datasets.combine_dataset_requirements import combine_dataset_requirements


class CombineDatasetRequirementsTestCase(unittest.TestCase):

    def test_run(self):
        expected_data = {
            "pbis_cost": [3, 12],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [5, 4],
                [5, 4]
            ],
            "dependencies": [None, None]
        }

        # load json file
        json_file = open("test_combination.json")
        json_data = json.load(json_file)

        actual_data = combine_dataset_requirements(json_data)

        self.assertEqual(expected_data["pbis_cost"], actual_data["pbis_cost"])
        self.assertEqual(expected_data["stakeholders_importances"], actual_data["stakeholders_importances"])
        self.assertEqual(expected_data["stakeholders_pbis_priorities"], actual_data["stakeholders_pbis_priorities"])
        self.assertEqual(expected_data["dependencies"], actual_data["dependencies"])

    def test_run2(self):
        expected_data = {
            "pbis_cost": [12, 7],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 5],
                [4, 5]
            ],
            "dependencies": [None, None]
        }

        # load json file
        json_file = open("test_combination2.json")
        json_data = json.load(json_file)

        actual_data = combine_dataset_requirements(json_data)

        self.assertEqual(expected_data["pbis_cost"], actual_data["pbis_cost"])
        self.assertEqual(expected_data["stakeholders_importances"], actual_data["stakeholders_importances"])
        self.assertEqual(expected_data["stakeholders_pbis_priorities"], actual_data["stakeholders_pbis_priorities"])
        self.assertEqual(expected_data["dependencies"], actual_data["dependencies"])

    def test_run3(self):
        expected_data = {
            "pbis_cost": [12, 7],
            "stakeholders_importances": [1, 3],
            "stakeholders_pbis_priorities": [
                [4, 5],
                [4, 5]
            ],
            "dependencies": [None, None]
        }

        # load json file
        json_file = open("test_combination3.json")
        json_data = json.load(json_file)

        actual_data = combine_dataset_requirements(json_data)

        self.assertEqual(expected_data["pbis_cost"], actual_data["pbis_cost"])
        self.assertEqual(expected_data["stakeholders_importances"], actual_data["stakeholders_importances"])
        self.assertEqual(expected_data["stakeholders_pbis_priorities"], actual_data["stakeholders_pbis_priorities"])
        self.assertEqual(expected_data["dependencies"], actual_data["dependencies"])
