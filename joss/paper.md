---
title: "cbcbeat: an adjoint-enabled framework for computational cardiac electrophysiology" 
tags:
- finite element method
- cardiac electrophysiology
- adjoint
authors:
- name: Marie E. Rognes 
  orcid: 0000-0002-6872-3710
  affiliation: 1
- name: Patrick E. Farrell
  orcid: 
  affiliation: 2
- name: Simon W. Funke
  orcid: 0000-0003-4709-8415
  affiliation: 1
- name: Johan E. Hake
  orcid: 0000-0002-4042-0128
  affiliation: 3
- name: Molly M. C. Maleckar
  orcid: 0000-0002-7012-3853
  affiliation: 4
affiliations:
 - name: Simula Research Laboratory
   index: 1
 - name: University of Oxford
   index: 2
 - name: Ski videregående skole
   index: 3
 - name: Allen Institute of Cell Science
   index: 4
date: 29 March 2017
bibliography: references.bib
---

# Summary

cbcbeat [@cbcbeat] is a Python-based software collection targeting
computational cardiac electrophysiology problems. cbcbeat contains
solvers of varying complexity and performance for the classical
monodomain and bidomain equations coupled with cardiac cell
models. The cbcbeat solvers are based on algorithms described in
[@SundnesEtAl2006] and the core FEniCS Project software
[@LoggEtAl2012]. All solvers allow for automated derivation and
computation of adjoint and tangent linear solutions, functional
derivatives and Hessians via the dolfin-adjoint software
[@FarrellEtAl2013]. The computation of functional derivatives in turn
allows for automated and efficient solution of optimization problems
such as those encountered in data assimillation or other inverse
problems.

The cbcbeat source code is hosted with Bitbucket
(https://bitbucket.org/meg/cbcbeat) with documentation on readthedocs
(http://cbcbeat.readthedocs.io).

# References 
