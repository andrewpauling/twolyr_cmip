twolyr_cmip
==============================
[![Build Status](https://github.com/andrewpauling/twolyr_cmip/workflows/Tests/badge.svg)](https://github.com/andrewpauling/twolyr_cmip/actions)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)[![pypi](https://img.shields.io/pypi/v/twolyr_cmip.svg)](https://pypi.org/project/twolyr_cmip)

Two-layer model code and fitting procedure

To install and use:

- Download the repo to NCAR machine and cd to directory:
```
git clone https://github.com/andrewpauling/twolyr_cmip.git
cd twolyr_cmip
```

- Install conda environment
```
conda env create -f environment.yml
python setup.py develop
```

- Download data to run the script
```
cd data
./getdata.sh
```


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-project">cookiecutter science project template</a>.</small></p>
