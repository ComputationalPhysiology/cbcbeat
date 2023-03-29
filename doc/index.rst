.. cbcbeat documentation master file, created by
   sphinx-quickstart on Fri Sep 12 10:27:56 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================================================
cbcbeat: an adjoint-enabled framework for computational cardiac electrophysiology
=================================================================================

cbcbeat is a Python-based software collection targeting computational
cardiac electrophysiology problems. cbcbeat contains solvers of
varying complexity and performance for the classical monodomain and
bidomain equations coupled with cardiac cell models. The cbcbeat
solvers are based on algorithms described in Sundnes et al (2006) and
the core FEniCS Project software (Logg et al, 2012). All cbcbeat
solvers allow for automated derivation and computation of adjoint and
tangent linear solutions, functional derivatives and Hessians via the
dolfin-adjoint software (Farrell et al, 2013). The computation of
functional derivatives in turn allows for automated and efficient
solution of optimization problems such as those encountered in data
assimillation or other inverse problems.

cbcbeat is based on the finite element functionality provided by the
FEniCS Project software, the automated derivation and computation of
adjoints offered by the dolfin-adjoint software and cardiac cell
models from the CellML repository.

cbcbeat originates from the `Center for Biomedical Computing
<http://cbc.simula.no>`__, a Norwegian Centre of Excellence, hosted by
`Simula Research Laboratory <http://www.simula.no>`__, Oslo, Norway.

Installation and dependencies:
==============================

The cbcbeat source code is hosted on GitHub:

  https://github.com/ComputationalPhysiology/cbcbeat

The cbcbeat solvers are based on the `FEniCS Project
<http://www.fenicsproject.org>`__ finite element library and its
extension `dolfin-adjoint <http://www.dolfin-adjoint.org>`__. Any type
of build of FEniCS and dolfin-adjoint should work, but cbcbeat has
mainly been developed using native source builds and is mainly tested
via Docker images.

See the separate file INSTALL in the root directory of the cbcbeat
source for a complete list of dependencies and installation
instructions.


Main authors:
=============

See the separate file AUTHORS in the root directory of the cbcbeat
source for the list of authors and contributors.

License:
========

cbcbeat is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

cbcbeat is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public
License along with cbcbeat. If not, see
<http://www.gnu.org/licenses/>.

Testing and verification:
=========================

The cbcbeat test suite is based on `pytest <http://pytest.org>`__ and
available in the test/ directory of the cbcbeat source. See the
INSTALL file in the root directory of the cbcbeat source for how to
run the tests.

cbcbeat uses GitHub issues for automated and continuous testing,
see the current test status of cbcbeat here:

  https://github.com/ComputationalPhysiology/cbcbeat/issues

Contributions:
==============

Contributions to cbcbeat are very welcome. If you are interested in
improving or extending the software please `contact us
<https://github.com/ComputationalPhysiology/cbcbeat>`__ via the issues or pull requests
on GitHub. Similarly, please `report
<https://github.com/ComputationalPhysiology/cbcbeat/issues>`__ issues via GitHub.

Documentation:
==============

The cbcbeat solvers are based on the Python interface of the `FEniCS
Project <http://www.fenicsproject.org>`__ finite element library and
its extension `dolfin-adjoint <http://www.dolfin-adjoint.org>`__. We
recommend users of cbcbeat to first familiarize with these
libraries. The `FEniCS tutorial
<https://fenicsproject.org/tutorial/>`__ and the `dolfin-adjoint
documentation
<https://www.dolfin-adjoint.org/en/latest/documentation/index.html>`__
are good starting points for new users.

Examples and demos:
-------------------

A collection of examples on how to use cbcbeat is available in the
demo/ directory of the cbcbeat source. We also recommend looking at
the test suite for examples of how to use the cbcbeat solvers.

API documentation:
------------------

.. toctree::
   :maxdepth: 2
   :numbered:

   cbcbeat

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
