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

    def test_add_bad_observation(self):
        data = np.zeros([ARBITRARY_ROWS, len(TEST_PARAMETERS)])
        frame = DataFrame(data, TEST_PARAMETERS)

        test_observation = []
        self.assertRaises(Exception, frame.add_observation, test_observation)

        test_observation = []
        for i, parameter in enumerate(TEST_PARAMETERS):
            test_observation.append(i+1)
        test_observation.pop()
        self.assertRaises(Exception, frame.add_observation, test_observation)

        test_observation = []
        for i, parameter in enumerate(TEST_PARAMETERS):
            test_observation.append(i+1)
        test_observation.append(i+1)
        self.assertRaises(Exception, frame.add_observation, test_observation)

    def test_add_observation(self):
        test_observation = []
        for i, parameter in enumerate(TEST_PARAMETERS):
            test_observation.append(i+1)

        data = np.zeros([ARBITRARY_ROWS, len(TEST_PARAMETERS)])
        frame = DataFrame(data, TEST_PARAMETERS).add_observation(test_observation)

        # Ensure observation row was added and number of parameters was unchanged
        self.assertEquals(ARBITRARY_ROWS+1, np.shape(frame)[0])
        self.assertEquals(len(TEST_PARAMETERS), np.shape(frame)[1])

        # Check observation was added successfully
        for i, row in enumerate(frame):
            for j, col in enumerate(frame[i]):
                if i == ARBITRARY_ROWS:
                    # If this is the new row
                    self.assertEquals(j+1, frame[i, j])
                else:
                    # Other rows should contain 0
                    self.assertEquals(0, frame[i, j])


if __name__ == '__main__':
    unittest.main()
