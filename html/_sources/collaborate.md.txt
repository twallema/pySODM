## Start to collaborate

When working on the model or the model code, make sure to read the following guidelines and keep them in mind. The main purpose of being rigid about the structure of the repository is to improve the general collaboration while keeping structure of the code in the long run and support reproducibility of the work. Still, it is work in progress and it should not block your work. In case you get stuck on something or you have a suggestion for improvement, feel free to open an [New Issue](https://github.com/UGentBiomath/COVID19-Model/issues/new) on the repository.

### Documentation website

Documentation consists of both the technical matter about the code as well as background information on the models. To keep these up to date and centralized, we use [Sphinx](https://www.sphinx-doc.org/en/master/) which enables us to keep the documentation together on a website.

The Sphinx setup provides the usage of both `.rst` file, i.e. [restructuredtext](https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html) as well as `.md` files, i.e. [Markdown](https://www.markdownguide.org/basic-syntax/). The latter is generally experienced as easier to write, while the former provides more advanced functionalities. Existing pages of the documentation reside in `~/docs/` and can be adjusted directly. In case you want to build the documentation locally, make sure you have the development dependencies installed (`pip install -e ".[develop]"`) to run the sphinx build script. To build locally, use,
```
python setup.py build_sphinx
```
When you want to create a new page, make sure to add the page to the `index.rst` in order to make the page part of the website. The website is build automatically using [github actions](REPLACE: https://github.com/twallema/COVID19-Model/blob/master/.github/workflows/deploy.yml#L22-L24) and the output is deployed to [https://ugentbiomath.github.io/COVID19-Model/](REPLACE: https://ugentbiomath.github.io/COVID19-Model/). The resulting html-website is created in the directory `build/html`. Double click any of the `html` file in the folder to open the website in your browser (no server required).

### The `pySODM` Python package

The code inside the `src/pySODM` directory is actually a Python package, which provides a number of additional benefits on the maintenance of the code.

Before doing any changes, always make sure your own version of your code (i.e. `fork`) is up to date with the `master` of the [main repository ](https://github.com/twallema/pySODM). First, check out the __[git workflow](./git_workflow.md)__ for a step-by-step explanation of the proposed workflow. For more information, see also:
- If you are a command line person, check [this workflow](https://gist.github.com/CristinaSolana/1885435)
- If you are not a command line person: [this workflow](https://www.sitepoint.com/quick-tip-sync-your-fork-with-the-original-without-the-cli/) can help you staying up to date.

For each of the functions you write, make sure to add the documentation to the function. We use the [numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) format to write documentation. For each function, make sure the following items are defined at least:

- Short summary (top line)
- Parameters
- Returns
- References (if applicable)

__Note:__ When adding new packages makes sure to update both,
- the environment file, [evironment.yml](https://github.com/twallema/pySODM/blob/master/environment.yml) for binder,
- the setup file, [setup.py](https://github.com/twallema/pySODM/blob/master/setup.py) file to include this dependency for the installation of the package.

### Repository layout overview

As the previous sections described, each subfolder of the repository has a specific purpose and we would ask to respect the general layout. Still, this is all work in progress, so alterations to it that improve the workflow are certainly possible. Please do your suggestion by creating a [New issue](https://github.com/twallema/pySODM/issues/new/choose).

__Remember:__ Anyone should be able to reproduce the final products with only the `code` in `src`!

#### code
```
├── src                                     <- all reusable code blocks
│   ├── pySODM
|   │   ├── models                          <- any code constructing the models
|   │   ├── optimization                    <- code related to parameter callibration
|   │   └── __init__.py                     <- structured as lightweight python package
│   ├── tests
|   │   ├── ... .py                         <- all test code during development
```

#### documentation
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

#### automate stuff
```
├── .github                                 <- Automate specific steps with github actions
│   ├── workflows
│   │   ├── deploy.yml
│   │   └── ...
```

#### other info
```
├── LICENSE
├── setup.py
└── README.md                               <- focus on how to get started, setup environment name conventions and
```
