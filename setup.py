# System imports
from distutils.core import setup
import platform
import sys
from os.path import join as pjoin

# Version number
major = 1
minor = 0

scripts = [pjoin("scripts", "gotran2beat"),
           pjoin("scripts", "gotran2dolfin"),
           ]

if platform.system() == "Windows" or "bdist_wininst" in sys.argv:
    # In the Windows command prompt we can't execute Python scripts
    # without a .py extension. A solution is to create batch files
    # that runs the different scripts.
    batch_files = []
    for script in scripts:
        batch_file = script + ".bat"
        f = open(batch_file, "w")
        f.write('python "%%~dp0\%s" %%*\n' % split(script)[1])
        f.close()
        batch_files.append(batch_file)
    scripts.extend(batch_files)

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
      )
