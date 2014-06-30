=====
Usage
=====

Classes and Readers
-------------------

Define Classes
~~~~~~~~~~~~~~

Frontier is used for classification based machine learning problems, to be useful
you must inform Frontier of the classes that define your problem.

For example, if you were to conduct an experiment to classify things that were
and were not owls, you might construct a dictionary akin to:

.. code-block:: python

    CLASSES = {
            "hoot": {
                "names": ["owl", "owls"],
                "code": 1,
            },
            "unhoot": {
                "names": ["cat", "dog", "pancake"],
                "code": 0,
            },
    }


Each key in the ``CLASSES`` dictionary names a particular class in your
classification problem. The values to the ``names`` and ``code`` keys refer to a
list of labels that belong to a particular class and the encoding used to refer to
that class respectively.


Define Readers
~~~~~~~~~~~~~~

Both data and targets are read in to Frontier via a ``Reader`` which you will
often need to implement yourself by inheriting from :class:`frontier.IO.AbstractReader`.

.. code-block:: python

    from frontier.IO.AbstractReader import AbstractReader

    class MyReader(AbstractReader):

        def __init__(self, filepath, CLASSES=None, auto_close=True):
            """Initialise the structures for storing data and construct the reader."""
            self.mydata = {
                "_id": "something" # Could use os.path.basename(filepath)
            }
            header_skip = 1 # Numer of initial lines to ignore
            super(MyReader, self).__init__(filepath, CLASSES, auto_close, header_skip)

        def process_line(self, line):
            """Process a line record in your file."""
            fields = line.split("\t")

            _id = fields[0]
            value = fields[1]

            self.data[_id] = value

        def get_data(self):
            """Interface to return read data."""
            return self.mydata

The name of the structure used to hold data is irrelevant, just that it is returned
sensibly by ``get_data``. It is expected (and required) that data readers will provide
an **_id** key in any data returned by ``get_data`` that corresponds to a key
in the structure returned by ``get_data`` of your chosen target reader (thus
linking a record of data to its target).


Import Readers
~~~~~~~~~~~~~~

These readers can then be used to automatically process data and targets from
single files or recursively through a directory. Currently it is expected that
targets will be enumerated in one file and data will be stored in some directory
structure.

.. code-block:: python

    from frontier import frontier
    from myreaders import DataReader, TargetReader

    data_dir = "/home/sam/Projects/owl_classifier/data/"
    target_path = "/home/sam/Projects/owl_classifier/targets.txt"

    statplexer = frontier.Statplexer(data_dir,
                                     target_path,
                                     CLASSES,
                                     DataReader,
                                     TargetReader)

The :class:`frontier.frontier.Statplexer` class will then read in data and target
inputs using the specified DataReader and TargetReader from the given paths.

The Statplexer can then be used to query the data and targets.


The Statplexer
--------------

Query Parameters or Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`frontier.frontier.Statplexer.list_parameters`
    Return a sorted list of all parameters

    .. code-block:: python

        ...
        parameters = statplexer.list_parameters()

:func:`frontier.frontier.Statplexer.find_parameters`
    Given a list of input strings, return a list of parameters which contain
    any of those strings as a substring

:func:`frontier.frontier.Statplexer.exclude_parameters`
    Given a list of input strings, return a list of parameters which do not
    contain any of the input strings as a substring, or if needed an exact
    match

    .. code-block:: python

        ...
        parameters = statplexer.exclude_parameters(["owl-ratio", "hoot"])


Retrieve Data and Target Pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`frontier.frontier.Statplexer.get_data_by_parameters`
    Return data for each observation, but only include columns
    for each parameter in the given list

:func:`frontier.frontier.Statplexer.get_data_by_target`
    Return data for each observation that have been classified in one of the
    targets specified and additionally only return columns for the
    parameters in the given list

    .. code-block:: python

        ...
        # Using the CLASSES above this would return data and targets for all data records
        # classified with code 1 (ie. in the "hoot" classification), limited to just the
        # "owl-ratio" and "hoot" parameters.
        data, target, levels = statplexer.get_data_by_target(["owl-ratio", "hoot"], 1)

