History
=======

0.1.3-dev
---------

* Introduced :class: `frontier.frame.DataFrame` which extends numpy's `ndarray`,
  that will now be returned from :class: `frontier.frontier.Statplexer`'s API
  functions that return read in data.
* `_test_variance` function of the `Statplexer` will now test the range of variance
  magnitudes across each parameter of read in data, issuing a warning and producing
  a table if a magnitude difference greater than ±1 is discovered.

0.1.2 (2014-08-12)
---------------------

* Fix `#2 <https://github.com/SamStudio8/frontier/issues/2>`_
    * Add `get_id` to data readers to prevent cluttering parameter space.
    * Update `TestBamcheckReader.test_id_key` to use `get_id()` instead of `get_data()["_id"]`

0.1.1 (2014-06-30)
---------------------

* Documentation now exists.
* Required data readers to specify an _id instead of forcing use of data file basename in the Statplexer.

0.1.0 (2014-06-28)
---------------------

* First release on PyPI.
