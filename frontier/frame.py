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
        self.frontier_labels = getattr(obj, 'frontier_labels', [])[:]
        self.frontier_label_index = getattr(obj, 'frontier_label_index', {}).copy()
        # We do not need to return anything

    def add_observation(self, observation_list):
        if len(observation_list) == len(self.frontier_label_index):
            # NOTE FUTURE(samstudio8)
            #      Somewhat inefficient to return a new DataFrame, perhaps create
            #      a wrapper around the DataFrame which can overwrite the frame
            #      without removing additional attributes like labels
            # NOTE FUTURE(samstiudio8)
            #      DataFrames that have undergone a transformation should provide
            #      a method for which to automatically transform new observations
            return DataFrame(np.vstack( (self, observation_list) ), self.frontier_labels)
        else:
            raise Exception("Number of parameters in frame does not match number of parameters given.")

    def exclude(self, exclude_labels):
        new_label_list = []
        index_list = []
        for label in self.frontier_labels:
            if label in exclude_labels:
                print("[NOTE] Dropping label '%s' from frame" % label)
            else:
                new_label_list.append(label)
                index_list.append(self.frontier_label_index[label])
        return DataFrame(self[:, index_list], new_label_list)

    # NOTE In a desire to avoid the perceived evils of `eval` I've chosen to just
    #      trust users to provide their own `lambda` functions to transform
    #      particular variables. In future we could use `pyparsing` to read in
    #      mathematical expressions in string form.
    # TODO Order could become an issue
    def transform(self, transformation_dict, add_unknown=False):
        transformed_self = self.copy()
        for key in transformation_dict:
            new_label = False
            if key not in self.frontier_label_index:
                if add_unknown:
                    transformed_self.frontier_labels.append(key)
                    transformed_self = DataFrame(
                            np.c_[transformed_self, np.zeros(np.shape(self)[0])],
                            transformed_self.frontier_labels)
                    new_label = True
                    print("[NOTE] Adding label '%s' to frame" % key)
                else:
                    raise Exception("Unknown label %s" % key)

            label_index = transformed_self.frontier_label_index[key]

            value = 0.0
            if not new_label:
                value = self[:, label_index]
            try:
                # NOTE Could use self here at the cost of reversibility
                #      self[:,self.frontier_label_index[key]] = transformation_dict[key](self[:,self.frontier_label_index[key]])
                transformed_self[:, label_index] = transformation_dict[key](value, self, None)

            except (TypeError, ValueError) as e:
                # TypeError
                # Try to apply the transformation to each applicable element
                # to support use of `math` module functions which require
                # scalars or length-1 arrays rather than lists.

                # ValueError
                # Try to apply the transformation to each applicable element
                # where a function that accepts an array is ambiguous for example
                #   max(f.get("A", i), f.get("B", i))
                # Are we looking at the maximum element in the array or not?
                # numpy will raise a ValueError which we can use to fall back
                # to applying the transformation to each individual row.
                try:
                    for i in range(0, np.shape(self)[0]):
                        value = 0.0
                        if not new_label:
                            value = self[i, label_index]
                        transformed_self[i, label_index] = transformation_dict[key](value, self, i)
                except TypeError as e:
                    raise Exception("TypeError '%s' encountered on key '%s'." % (e, key))
        return DataFrame(transformed_self, transformed_self.frontier_labels)

    def get_single(self, label, i):
        # Raise a ValueError to force a fallback to row-by-row transformations
        if i is None:
            raise ValueError()
        return self.get(label, i)

    def get(self, label, i=None):
        if label in self.frontier_label_index:
            index = self.frontier_label_index[label]
        else:
            raise Exception("Label %s not in DataFrame" % label)

        if i is not None:
            return self[:, index][i]
        return self[:, index]

