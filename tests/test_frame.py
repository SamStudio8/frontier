import numpy as np
import unittest

from frontier.frame import DataFrame

TEST_PARAMETERS = [
    "owl-ratio",
    "wing-span",
    "wiseness-coefficient",
    "talon-sharpness",
    "talon-length",
    "number-of-doctorates",
]
ARBITRARY_ROWS = 10

class TestFrame(unittest.TestCase):

    # Ensure labels were added to columns in the order provided and
    # are correctly indexed
    def test_frame_label(self):
        data = np.empty([ARBITRARY_ROWS, len(TEST_PARAMETERS)])
        frame = DataFrame(data, TEST_PARAMETERS)

        for i, name in enumerate(TEST_PARAMETERS):
            self.assertEquals(i, frame.frontier_label_index[name])
            self.assertEquals(name, frame.frontier_labels[i])

    # Exceptions should be raised if there are too few or too many labels
    def test_frame_label_invalid_length(self):
        data = np.empty([ARBITRARY_ROWS, len(TEST_PARAMETERS)-1])
        self.assertRaises(Exception, DataFrame, data, TEST_PARAMETERS)

        data = np.empty([ARBITRARY_ROWS, len(TEST_PARAMETERS)+1])
        self.assertRaises(Exception, DataFrame, data, TEST_PARAMETERS)

    # Exception should be raised if a label is duplicated
    def test_frame_label_duplicate(self):
        TEST_PARAMETERS_COPY = TEST_PARAMETERS[:]
        TEST_PARAMETERS_COPY.append(TEST_PARAMETERS_COPY[0])
        data = np.empty([ARBITRARY_ROWS, len(TEST_PARAMETERS_COPY)])
        self.assertRaises(Exception, DataFrame, data, TEST_PARAMETERS_COPY)
