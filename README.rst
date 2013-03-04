.. -*- mode: rst -*-

OkStereotypeMe!
==============

Source code for okstereotype.me

Dependencies
~~~~~~~~~~~~

- scikit-learn https://github.com/scikit-learn/scikit-learn
- django https://www.djangoproject.com/
- rabbitmq http://www.rabbitmq.com/
- pandas http://pandas.pydata.org/
- Amueller's word cloud https://github.com/amueller/word_cloud
- XKCD plot for matplotlib http://jakevdp.github.com/blog/2012/10/07/xkcd-style-plots-in-matplotlib/

Usage
~~~~~

First scrape data from okcupid and store in a pandas dataframe as described in ``pick_best_model_v1.py`` and test how different
models perform on your dataset.

Second, pickle the data using ``pick_best_model_v2.py``

Third, run the django server using ``manage.py`` and run the rpc server using ``queue/rpc_server.py``

- ``pick_best_model_v1.py`` should only be used for testing purposes
- ``multi_q.py`` is not a good way to create a multicore vectorization of new essays.  Use ``rpc_server.py`` instead.
