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

# TODO Test DataFrame.get
class TestFrame(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialise a data frame with trivial data
        cls.data = np.zeros([ARBITRARY_ROWS, len(TEST_PARAMETERS)])
        cls.frame = DataFrame(cls.data, TEST_PARAMETERS)
        for i in range(0, ARBITRARY_ROWS):
            for j in range(0, len(TEST_PARAMETERS)):
                cls.data[i, j] = (i+1)*(j+1)

    # TODO Test stub for copying DataFrame
    def test_copy(self):
        frame_copy = self.frame.copy()

        # Ensure labels match original frame metadata
        for label in self.frame.frontier_labels:
            self.assertIn(label, frame_copy.frontier_labels)
            self.assertIn(label, frame_copy.frontier_label_index)

        # Ensure label index is correct
        for label in self.frame.frontier_label_index:
            self.assertEqual(label, frame_copy.frontier_labels[frame_copy.frontier_label_index[label]])

        # Ensure metadata objects are different (and not pointers to the same structures)
        self.assertNotEqual(id(frame_copy.frontier_labels), id(self.frame.frontier_labels))
        self.assertNotEqual(id(frame_copy.frontier_label_index), id(self.frame.frontier_label_index))

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
        frame = self.frame

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

        num_tests = 0
        transformed_frame = frame.transform(test_transformations)
        for i, row in enumerate(transformed_frame):
            for j, col in enumerate(transformed_frame[i]):
                if j in transform_map:
                    num_tests += 1
                    self.assertEquals(transform_map[j](frame[i, j], frame, j),
                                        transformed_frame[i, j])
                else:
                    # Check nothing was transformed if it shouldn't have been
                    num_tests += 1
                    self.assertEquals(frame[i, j], transformed_frame[i, j])
        self.assertEquals(np.shape(transformed_frame)[0] * np.shape(transformed_frame)[1], num_tests)

    def test_bad_transform(self):
        frame = self.frame
        self.assertRaises(Exception, frame.transform, { TEST_PARAMETERS[0]: "hoot" })

    def test_transform_with_other_labels(self):
        frame = self.frame

        transform_map = {
            0: lambda x, f, i: x + f.get(TEST_PARAMETERS[0], i), # Increase owl capacity
            8: lambda x, f, i: np.mean(f.get(TEST_PARAMETERS[8], None)) - x  # Normalise the splines
        }
        test_transformations = {}
        for label_index in transform_map:
            test_transformations[TEST_PARAMETERS[label_index]] = transform_map[label_index]

        num_tests = 0
        transformed_frame = frame.transform(test_transformations)
        for i, row in enumerate(transformed_frame):
            for j, col in enumerate(transformed_frame[i]):
                if j in transform_map:
                    num_tests += 1
                    self.assertEquals(transform_map[j](frame[i, j], frame, i),
                                        transformed_frame[i, j])
                else:
                    # Check nothing was transformed if it shouldn't have been
                    num_tests += 1
                    self.assertEquals(frame[i, j], transformed_frame[i, j])
        self.assertEquals(np.shape(transformed_frame)[0] * np.shape(transformed_frame)[1], num_tests)

    def test_transform_mix_array_and_scalar_functionality(self):
        frame = self.frame

        transform_map = {
            0: lambda x, f, i: f.get(TEST_PARAMETERS[0], i) + math.factorial(math.ceil(x/2)), # Factorialising owl capacity
        }
        test_transformations = {}
        for label_index in transform_map:
            test_transformations[TEST_PARAMETERS[label_index]] = transform_map[label_index]

        num_tests = 0
        transformed_frame = frame.transform(test_transformations)
        for i, row in enumerate(transformed_frame):
            for j, col in enumerate(transformed_frame[i]):
                if j in transform_map:
                    num_tests += 1
                    self.assertEquals(transform_map[j](frame[i, j], frame, i),
                                        transformed_frame[i, j])
                else:
                    # Check nothing was transformed if it shouldn't have been
                    num_tests += 1
                    self.assertEquals(frame[i, j], transformed_frame[i, j])
        self.assertEquals(np.shape(transformed_frame)[0] * np.shape(transformed_frame)[1], num_tests)

    def test_transform_ambiguous_function(self):
        frame = self.frame

        transform_map = {
            0: lambda x, f, i: max([f.get(TEST_PARAMETERS[0], i), f.get(TEST_PARAMETERS[6], i)]),
        }
        test_transformations = {}
        for label_index in transform_map:
            test_transformations[TEST_PARAMETERS[label_index]] = transform_map[label_index]

        print frame
        num_tests = 0
        transformed_frame = frame.transform(test_transformations)
        print transformed_frame
        for i, row in enumerate(transformed_frame):
            for j, col in enumerate(transformed_frame[i]):
                if j in transform_map:
                    num_tests += 1
                    self.assertEquals(transform_map[j](frame[i, j], frame, i),
                                        transformed_frame[i, j])
                else:
                    # Check nothing was transformed if it shouldn't have been
                    num_tests += 1
                    self.assertEquals(frame[i, j], transformed_frame[i, j])
        self.assertEquals(np.shape(transformed_frame)[0] * np.shape(transformed_frame)[1], num_tests)

    def test_transform_new_label(self):
        frame = self.frame

        # NOTE We are not restricted to using an integer to label the new variable,
        #      it just plays nicely with the assertion step later on...
        ARBITRARY_NAME = len(TEST_PARAMETERS) # New index
        transform_map = {
            ARBITRARY_NAME: lambda x, f, i: f.get(TEST_PARAMETERS[1], i) * f.get(TEST_PARAMETERS[2], i)
        }
        test_transformations = {}
        for label in transform_map:
            test_transformations[label] = transform_map[label]

        num_tests = 0
        transformed_frame = frame.transform(test_transformations, add_unknown=True)
        for i, row in enumerate(transformed_frame):
            for j, col in enumerate(transformed_frame[i]):
                if j in transform_map:
                    num_tests += 1
                    self.assertEquals(transform_map[j](0.0, frame, i),
                                        transformed_frame[i, j])
                else:
                    # Check nothing was transformed if it shouldn't have been
                    num_tests += 1
                    self.assertEquals(frame[i, j], transformed_frame[i, j])
        self.assertEquals(np.shape(transformed_frame)[0] * np.shape(transformed_frame)[1], num_tests)

        # Check label was appended successfully
        self.assertEquals(ARBITRARY_NAME, transformed_frame.frontier_labels[-1])
        self.assertEquals(ARBITRARY_NAME, transformed_frame.frontier_label_index[ARBITRARY_NAME])

    def test_exclude(self):
        frame = self.frame
        EXCLUDE_LABEL_INDEXES = [3, 8]
        EXCLUDE_LABELS = []
        for i in EXCLUDE_LABEL_INDEXES:
            EXCLUDE_LABELS.append(TEST_PARAMETERS[i])
        transformed_frame = frame.exclude(EXCLUDE_LABELS)

        # Check frame shape
        self.assertEquals(np.shape(frame)[0], np.shape(transformed_frame)[0])
        self.assertEquals(np.shape(frame)[1]-len(EXCLUDE_LABELS), np.shape(transformed_frame)[1])

        # Ensure data dropped is actually missing
        num_tests = 0
        for i, row in enumerate(frame):
            offset = 0
            for j, col in enumerate(frame[i]):
                if j in EXCLUDE_LABEL_INDEXES:
                    # Try to find the unique element of the original frame indexed
                    # by the current i,j anywhere in the current transformed row
                    self.assertNotIn(frame[i, j], transformed_frame[i])

                    # Track the number of columns missing by this j
                    offset += 1
                    num_tests += 1
                else:
                    # Ensure undropped data is unchanged
                    self.assertEqual(frame[i, j], transformed_frame[i, j - offset])
                    num_tests += 1
        self.assertEquals(np.shape(frame)[0] * np.shape(frame)[1], num_tests)

        # Ensure label does not appear in frame metadata
        for excluded_label in EXCLUDE_LABELS:
            self.assertNotIn(excluded_label, transformed_frame.frontier_labels)
            self.assertNotIn(excluded_label, transformed_frame.frontier_label_index)

        # Ensure label index is correct
        for label in transformed_frame.frontier_label_index:
            self.assertEqual(label, transformed_frame.frontier_labels[transformed_frame.frontier_label_index[label]])


if __name__ == '__main__':
    unittest.main()
