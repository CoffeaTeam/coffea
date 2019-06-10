.. coffea documentation master file, created by
   sphinx-quickstart on Fri Jun  7 09:23:02 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

coffea - Columnar Object Framework For Efficient Analysis
=========================================================

coffea is a prototype package for pulling together all the typical needs
of a high-energy collider physics (HEP) experiment analysis using the scientific
python ecosystem. It makes use of `uproot <https://github.com/scikit-hep/uproot>`_
and `awkward\\-array <https://github.com/scikit-hep/awkward-array>`_ to provide an 
array\\-based syntax for manipulating HEP event data in an efficient and numpythonic 
way. There are  sub\\-packages that implement histogramming, plotting, and look\\-up
table functionalities that are needed to convey scientific insight, apply transformations 
to data, and correct for discrepancies in Monte Carlo simulations compared to data.

coffea also supplies facilities for horizontally scaling an analysis in order to reduce 
time\\-to\\-insight in a way that is largely independent of the resource the analysis
is being executed on. By making use of modern *big\\-data* technologies like 
`Apache Spark <https://spark.apache.org/>`_ and `parsl <https://github.com/Parsl/parsl>`_
it is possible with coffea to scale a HEP analysis from a testing on a laptop to: a large
multi\\-core server, computing clusters, and super-computers without the need to alter or
otherwise adapt the analysis code itself.

coffea is a HEP community project collaborating with `iris\\-hep <http://iris-hep.org/>`_ 
and a currently a prototype. We welcome input to improve its quality as we progress towards
a sensible refactorization into the scientific python ecosystem and a first release. Please
feel free to contribute at our `github repo <https://github.com/CoffeaTeam/coffea>`_!


.. toctree::

   quick_start
   coffea-introduction.ipynb
   reference
..
   userguide/index
   faq
   devguide/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
