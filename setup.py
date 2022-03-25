# System imports
from distutils.core import setup
import platform
import sys
from os.path import join as pjoin

# Version number
major = 2019
minor = 1.2

requirements = []
extras_require={
    "gotran": ["gotran"],
    "goss": ["pygoss"],
    "test": ["pytest", "matplotlib", "pytest-cov"],
    "docs": ["sphinx", "sphinx_book_theme"]
}
extras_require.update(
    {"all": [val for values in extras_require.values() for val in values]}
)
scripts = [pjoin("scripts", "gotran2beat"),
           pjoin("scripts", "gotran2dolfin"),
           ]

with open("README.rst") as readme_file:
  readme = readme_file.read()


setup(name = "cbcbeat",
      version = "{0}.{1}".format(major, minor),
      description = """
      An adjoint-enabled framework for computational cardiac electrophysiology
      """,
      author = "M. E. Rognes, J. E. Hake, P. E. Farrell, S. W. Funke",
      author_email = "meg@simula.no",
      packages = ["cbcbeat", "cbcbeat.cellmodels",],
      package_dir = {"cbcbeat": "cbcbeat"},
      scripts = scripts,
      long_description=readme,
      include_package_data=True,
      install_requires=requirements,
      url="https://github.com/ComputationalPhysiology/cbcbeat",
      extras_require=extras_require,
      project_urls={
          "Source": "https://github.com/ComputationalPhysiology/cbcbeat",
      },
      zip_safe=False,
    )
