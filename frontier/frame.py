import numpy as np

# DataFrame
#   Comments and help on the proper use of __new__ and __array_finalize__ and
#   subclassing ndarray in general located on the scipy documentation at:
#       docs.scipy.org/doc/numpy/user/basics.subclassing.htm
class DataFrame(np.ndarray):

    def __new__(cls, input_array, labels):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # Add the label list to the array, ensuring the number of labels
        # matches the number of columns
        # TODO Catch an IndexError if the array shape is (1,1) ?
        if obj.shape[1] == len(labels):
            obj.frontier_labels = labels

            obj.frontier_label_index = {}
            for i, label in enumerate(labels):
                if label in obj.frontier_label_index:
                    raise Exception("Duplicate label '%s' encountered when labelling DataFrame." % label)
                obj.frontier_label_index[label] = i
        else:
            raise Exception("Number of labels did not match number of columns.")

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(DataFrame, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. DataFrame():
        #    obj is None
        #    (we're in the middle of the DataFrame.__new__
        #    constructor, and self.info will be set when we return to
        #    DataFrame.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(DataFrame):
        #    obj is arr
        #    (type(obj) can be DataFrame)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is DataFrame
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set any default values, because this
        # method sees all creation of default objects - with the
        # DataFrame.__new__ constructor, but also with
        # arr.view(DataFrame).
        self.frontier_labels = getattr(obj, 'frontier_labels', [])
        self.frontier_label_index = getattr(obj, 'frontier_labels', {})
        # We do not need to return anything

    def add_observation(self, observation_list):
        if len(observation_list) == len(self.frontier_label_index):
            # NOTE FUTURE(samstudio8)
            #      Somewhat inefficient to return a new DataFrame, perhaps create
            #      a wrapper around the DataFrame which can overwrite the frame
            #      without removing additional attributes like labels
            return DataFrame(np.vstack( (self, observation_list) ), self.frontier_labels)
        else:
            raise Exception("Number of parameters in frame does not match number of parameters given.")

    def exclude(self, labels):
        index_list = []
        for label in labels:
            if label in obj.frontier_label_index:
                index_list.append(obj.frontier_label_index[label])
            else:
                print("[WARN] Label %s not in DataFrame" % label)

        return self[:, index_list]

    def multiply_by_label(self, multiplier, label):
        if label in self.frontier_label_index:
            self[:,self.frontier_label_index[label]] *= multiplier
        else:
            raise Exception("Unknown label %s" % label)
