from setuptools import setup, find_packages
from codecs import open

setup(
  name='TPCwithDNN',

  # LAST-TAG is a placeholder. Automatically replaced at deploy time with the right tag
  version='LAST-TAG',

  description='TPCwithDNN',

  # Long description from Markdown -- https://dustingram.com/articles/2018/03/16/markdown-descriptions-on-pypi
  # Filter out lines that look like GitHub "badges"
  long_description="\n".join([ line for line in open('README.md').read().split("\n") if not line.startswith("[![") ]),
  long_description_content_type='text/markdown',

  url='https://github.com/AliceO2Group/TPCwithDNN',
  #author='AUTHOR',
  #author_email='AUTHOR-EMAIL',
  #license='LICENSE',
  classifiers=[

    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',

    # Indicate who your project is intended for
    #'Intended Audience :: Education',
    'Topic :: Scientific/Engineering :: Physics',

    # Pick your license as you wish (should match "license" above)
    #'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 3.6',
    ],

  # What does your project relate to?
  #keywords='KEYWORDS',

  # You can just specify the packages manually here if your project is
  # simple. Or you can use find_packages().
  packages=find_packages(),

  # Alternatively, if you want to distribute just a my_module.py, uncomment
  # this:
  #   py_modules=["my_module"],

  # List run-time dependencies here.  These will be installed by pip when
  # your project is installed. For an analysis of "install_requires" vs pip's
  # requirements files see:
  # https://packaging.python.org/en/latest/requirements.html
  install_requires=[ "numpy==1.18.5", "pandas", "matplotlib", "scipy==1.4.1",
                     "keras==2.3.1", "PyYaml", "RootInteractive", "tf-nightly-gpu==2.4.0.dev20200908",
                     "pydot", "pylint"],

  python_requires='>=3.6, <3.7',

  # List additional groups of dependencies here (e.g. development
  # dependencies). You can install these using the following syntax,
  # for example:
  # $ pip install -e .[dev,test]
  extras_require={
  },

  # If there are data files included in your packages that need to be
  # installed, specify them here. Note that you need to specify those files in
  # MANIFEST.in as well, since Python tools behave inconsistently
  include_package_data=False,
  #package_data={ "tpcwithdnn.data1": [ "list",
  #                                     "of",
  #                                     "package",
  #                                     "data" ],
  #               "tpcwithdnn.data2": [ "more",
  #                                     "data" ] },

  # Although 'package_data' is the preferred approach, in some case you may
  # need to place data files outside of your packages. See:
  # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
  # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
  data_files=[],

  # To provide executable scripts, use entry points in preference to the
  # "scripts" keyword. Entry points provide cross-platform support and allow
  # pip to create the appropriate form of executable for the target platform.
  # See: https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/
  #entry_points={
  #    "console_scripts": [ "entry-point1 = tpcwithdnn.some-script:function",
  #                         "entry-point2= tpcwithdnn.other-script:function" ] }
)
