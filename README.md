# Ruggles Lab Member Handy Code: `ruggles` package

* Each module named for the member that contributes

## Installation

```bash
pip install git+https://github.com/ruggleslab/ruggles
```

The code above requires `pip` and `git`.


## Usage


### Python
To access the function `easy_subplots` from the `grant.utils` module, for example, import the module and access the function like any other package:

```python
from ruggles import grant.utils as gu

fig, axes = gu.easy_subplots(ncols=2, nrows=5)
```

### R
To access the R code, don't. Let me think about this first. (Will ask members during this presentations.)


## Contributing

### Setting up your module
* Ask Grant, they'll create a folder and adjust the `setup.py`
* Contribute code as you like to your named folder


### Docstrings
* Writing useful docstrings will help members out a lot in using your code!
