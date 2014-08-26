import math
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
    "voles-vanquished",
    "best-altitude",
    "splines"
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

    def test_transform(self):
        # Initialise data frame with trivial data
        data = np.zeros([ARBITRARY_ROWS, len(TEST_PARAMETERS)])
        frame = DataFrame(data, TEST_PARAMETERS)
        for i in range(0, ARBITRARY_ROWS):
            for j in range(0, len(TEST_PARAMETERS)):
                data[i, j] = (i+1)*(j+1)

        transform_map = {
            0: lambda x, f, i: x**2, # Releasing owls
            1: lambda x, f, i: x-1, # Clipping wings
            2: lambda x, f, i: x+10,# Reading books
            3: lambda x, f, i: x/2, # Manicuring
            4: lambda x, f, i: x-2, # More manicuring
            5: lambda x, f, i: x+1, # Awarding degrees
            6: lambda x, f, i: math.exp(x), # Catch some dinner
          # 7                            No flying today.
            8: lambda x, f, i: math.sqrt(x) # Doing something with splines
        }
        test_transformations = {}
        for label_index in transform_map:
            test_transformations[TEST_PARAMETERS[label_index]] = transform_map[label_index]

        transformed_frame = frame.transform(test_transformations)
        for transform in test_transformations:
            for i, row in enumerate(frame):
                for j, col in enumerate(frame[i]):
                    if j in transform_map:
                        self.assertEquals(transform_map[j](frame[i, j], frame, j),
                                          transformed_frame[i, j])
                    else:
                        # Check nothing was transformed if it shouldn't have been
                        self.assertEquals(frame[i, j], transformed_frame[i, j])

    def test_bad_transform(self):
        # Initialise data frame with trivial data
        data = np.zeros([ARBITRARY_ROWS, len(TEST_PARAMETERS)])
        frame = DataFrame(data, TEST_PARAMETERS)
        for i in range(0, ARBITRARY_ROWS):
            for j in range(0, len(TEST_PARAMETERS)):
                data[i, j] = (i+1)*(j+1)

        self.assertRaises(Exception, frame.transform, { TEST_PARAMETERS[0]: "hoot" })

    def test_transform_with_other_labels(self):
        # Initialise data frame with trivial data
        data = np.zeros([ARBITRARY_ROWS, len(TEST_PARAMETERS)])
        frame = DataFrame(data, TEST_PARAMETERS)
        for i in range(0, ARBITRARY_ROWS):
            for j in range(0, len(TEST_PARAMETERS)):
                data[i, j] = (i+1)*(j+1)

        transform_map = {
            0: lambda x, f, i: x + f.get("owl-ratio", i), # Increase owl capacity
            8: lambda x, f, i: np.mean(f.get("owl-ratio", None)) - x  # Normalise the splines
        }
        test_transformations = {}
        for label_index in transform_map:
            test_transformations[TEST_PARAMETERS[label_index]] = transform_map[label_index]

        transformed_frame = frame.transform(test_transformations)
        for transform in test_transformations:
            for i, row in enumerate(frame):
                for j, col in enumerate(frame[i]):
                    if j in transform_map:
                        self.assertEquals(transform_map[j](frame[i, j], frame, i),
                                          transformed_frame[i, j])
                    else:
                        # Check nothing was transformed if it shouldn't have been
                        self.assertEquals(frame[i, j], transformed_frame[i, j])

if __name__ == '__main__':
    unittest.main()
