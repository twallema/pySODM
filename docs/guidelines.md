## Contribution guidelines

### `pySODM` Repository structure

The code inside the `~/src/pySODM` directory is the actual Python package, while the code inside the `~/tutorials` directory provides some examples of applications of the `pySODM` package. The `pySODM` environment (lives inside `~/environement.yml`) groups the necessary dependencies to run the tutorials in a convenient conda environment. Each subfolder of the repository has a specific purpose and we would ask to respect the general layout. Still, this is all work in progress, so alterations to it that improve the workflow are certainly possible. Please do your suggestion by creating a [New issue](https://github.com/twallema/pySODM/issues/new/choose).

__Remember:__ Anyone should be able to reproduce the final products with the code in `~/src` and the dependencies listed in `~/setup.py`!

#### Code
```
├── src                                     <- all reusable code blocks
│   ├── pySODM
|   │   ├── models                          <- code related to constructing models
|   │   ├── optimization                    <- code related to parameter callibration
|   │   └── __init__.py                     <- structured as lightweight python package
│   ├── tests
|   │   ├── ... .py                         <- all test code during development
```

#### Documentation
```
├── docs                                    <- documentation
│   ├── conf.py
│   ├── index.rst                           <- explanations are written inside markdown or st files
│   ├── ... .md                             <- pages of the documentation website
│   ├── Makefile
│   └── _static
│   │   └── figs                            <- figs linked to the documentation
│   │       ├── ...
```

#### Automate stuff
```
├── .github                                 <- Automate specific steps with github actions
│   ├── workflows
│   │   ├── deploy.yml
│   │   └── ...
```

#### Other files
```
├── environment.yml                         <- A conda environment bundling all dependencies to run the tutorials
├── LICENSE                                 <- MIT License
├── setup.py                                <- Package name, version and dependencies
└── README.md                               <- focus on how to get started, setup environment name conventions and
```

### Dependencies

Always keep in mind additional dependencies lower `pySODM`'s life expectancy. Consider all alternatives before adding a dependency to `pySODM`! When adding a new dependency to `pySODM`'s base code (lives inside `~/src/`) be sure to add the dependency to the setup file [setup.py](https://github.com/twallema/pySODM/blob/master/setup.py). Need a dependency for a tutorial only (not needed in `~/src/`)? Add it to the [evironment file](https://github.com/twallema/pySODM/blob/master/environment.yml) only.

### Tests

Three test scripts are defined in `~/src/tests/`: 1) `test_ODEModel.py` to test the initializiation and simulation of ODE models, 2) `test_SDEModel.py`, similar but for SDE models and 3) `test_calibration.py`, to test the common calibration workflow. Testing these routines is usefull to verify that modifications made to the `pySODM` code do not break code. Or alternatively, if the code does break, to see where it breaks. The test routines must pass when performing a PR to Github. To run a test locally,
```
conda activate pySODM
pytest test_calibration.py
```

### Documentation

#### Website

Documentation consists of both the technical matter about the code as well as background information on the models. To keep these up to date and centralized, we use [Sphinx](https://www.sphinx-doc.org/en/master/) which enables us to keep the documentation together on a website.

The Sphinx setup provides the usage of both `.rst` file, i.e. [restructuredtext](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html) as well as `.md` files, i.e. [Markdown](https://www.markdownguide.org/basic-syntax/). The latter is generally experienced as easier to write, while the former provides more advanced functionalities. Existing pages of the documentation reside in `~/docs/` and can be adjusted directly. In case you want to build the documentation locally, make sure you have the development dependencies installed (`pip install -e ".[develop]"`) to run the sphinx build script. To build locally, use,
```
python setup.py build_sphinx
```
When you want to create a new page, make sure to add the page to the `index.rst` in order to make the page part of the website. The website is build automatically using [github actions](https://github.com/twallema/pySODM/blob/master/.github/workflows/deploy.yml#L22-L24) and the output is deployed to [https://twallema.github.io/pySODM/](https://twallema.github.io/pySODM/). The resulting html-website is created in the directory `build/html`. Double click any of the `html` file in the folder to open the website in your browser (no server required).

#### Docstrings

For each function or class you write, make sure to add documentation to the function. We use the [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) format to write documentation. For each function, make sure the following items are defined at least:

- Short summary: What does the function or class do? (top line)
- Parameters: For every input variable, list both the type and a short description of what it represents.
- Returns: For every output variable, list both the type and a short description of what it represents.
- References (if applicable)

As an example, consider,

```
def add(a, b):
   """The sum of two numbers.

    Parameters
    ==========

    a: float
        The first number.
    b: float
        The second number.

    Returns
    =======

    sum: float
        The sum of the first and second number.
   """

    sum = a + b

   return sum
```

### Coding guidelines

The following are some guidelines on how new code should be written. Of course, there are special cases and there will be exceptions to these rules. However, uniformly formatted code makes it easier to share code ownership. The `pydov` project tries to closely follow the official Python guidelines detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/) which detail how code should be formatted and indented. Please read it and follow it.

In addition, we add the following guidelines:

* DRY: [Don't Repeat Yourself](https://www.plutora.com/blog/understanding-the-dry-dont-repeat-yourself-principle)
* Use underscores to separate words in non class names: `n_samples` rather than `nsamples`.
* Avoid multiple statements on one line. Prefer a line return after a control flow statement (`if/for`).
* Please don’t use `import *` in any case. It is considered harmful by the official Python recommendations. It makes the code harder to read as the origin of symbols is no longer explicitly referenced, but most important, it prevents using a static analysis tool like pyflakes to automatically find bugs.
* Provide a [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) for all your functions and classes.
* Please use lowercase names (eventually with `_`) as column names for pandas Dataframes.

### Git Workflow

Before doing any changes, always make sure your own version of your code (i.e. `fork`) is up to date with the `master` of the [main repository ](https://github.com/twallema/pySODM). First, check out the __[git workflow](./git_workflow.md)__ for a step-by-step explanation of the proposed workflow. For more information, see also,
- If you are a command line person, check [this workflow](https://gist.github.com/CristinaSolana/1885435)
- If you are not a command line person: [this workflow](https://www.sitepoint.com/quick-tip-sync-your-fork-with-the-original-without-the-cli/) can help you staying up to date.
