# Builds a Docker image with OpenTidalFarm master
# version built from gitsources. It is based on
# the dev-dolfin-adjoint image available at
#
#      quay.io/dolfinadjoint/dev-dolfin-adjoint
#
# Authors:
# Simon Funke <simon@simula.no>

FROM quay.io/dolfinadjoint/dolfin-adjoint:dolfin-adjoint-2017.2.0
MAINTAINER Simon Funke <simon@simula.no>

USER root

RUN apt-get update && apt-get install -y \
    mercurial

RUN hg clone https://bitbucket.org/meg/cbcbeat
RUN cd cbcbeat; pip install .

USER fenics
COPY WELCOME $FENICS_HOME/WELCOME

USER root
