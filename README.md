# cbcbeat: an adjoint-enabled framework for computational cardiac electrophysiology

[![image](https://codecov.io/gh/ComputationalPhysiology/cbcbeat/branch/master/graph/badge.svg?token=Z4BCNQRRPU)](https://codecov.io/gh/ComputationalPhysiology/cbcbeat)

[![image](https://circleci.com/gh/ComputationalPhysiology/cbcbeat.svg?style=shield)](https://circleci.com/gh/ComputationalPhysiology/cbcbeat)

cbcbeat is a collection of Python-based solvers for cardiac
electrophysiology models. cbcbeat offers basic and optimized solvers for
the bidomain and monodomain equations coupled with cardiac cell models.
cbcbeat is based on the FEniCS Project and dolfin-adjoint.

For more information visit https://computationalphysiology.github.io/cbcbeat/

## Installation:

See separate file ./INSTALL for how to install the cbcbeat module and a
list of dependencies.

## Documentation:

The cbcbeat documention is availble on readthedocs, see:

> https://computationalphysiology.github.io/cbcbeat/

For manually updating the API and demo documentation, run:

> cd doc make html

assuming that cbcbeat and the optional dependency Sphinx
(sphinx-doc.org) is installed.

## Automated Testing:

cbcbeat uses GitHub Actions for automated and continuous testing:

> https://github.com/ComputationalPhysiology/cbcbeat/actions

## License:

cbcbeat is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

cbcbeat is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see \<<http://www.gnu.org/licenses/>\>.
