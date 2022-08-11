# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

import os
import sys
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by Megatron.")

MAJOR = 1
MINOR = 1.5

# Use the following formatting: (major, minor)
VERSION = (MAJOR, MINOR)

__version__ = '.'.join(map(str, VERSION)) + '.bs'
__package_name__ = 'megatron-lm'
__contact_names__ = 'NVIDIA INC'
__url__ = 'https://github.com/NVIDIA/Megatron-LM'
__download_url__ = 'https://github.com/NVIDIA/Megatron-LM/releases'
__description__ = 'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.'
__license__ = 'See https://github.com/NVIDIA/Megatron-LM/blob/master/LICENSE'
__keywords__ = 'deep learning, Megatron, gpu, NLP, nvidia, pytorch, torch, language'

with open("README.md", "r") as fh:
    long_description = fh.read()

###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content if not x.strip().startswith("#") and not x.strip() == ""]


install_requires = req_file(
    "requirements/requirements.txt"
) + req_file(
    "requirements/requirements_dev.txt"
)
setuptools.setup(
    name=__package_name__,
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,
    description=__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url=__url__,
    author=__contact_names__,
    maintainer=__contact_names__,
    # The licence under which the project is released
    license=__license__,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        # Indicate what your project relates to
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        # Supported python versions
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        # Additional Setting
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
    #python_requires=">=3.6",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    keywords=__keywords__,
)
