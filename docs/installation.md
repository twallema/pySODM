## Installation

### Get the code

- Create a [`github`](https://github.com/) account if you do not have one already.
- On the [pySODM Github repository page](https://github.com/twallema/pySODM) click the `Fork` button.
- From your own repository page (your account) of `pySODM`, use [`git`](https://git-scm.com/) to download the code to your own computer. See the [Github documentation](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) on how to clone/download a repository.

When all went fine, you should have the code on your computer in a directory called `pySODM`.

### Install Python (conda) and packages

To use the code, make sure you have Python (conda) and the required dependency packages installed. We recommend using `Anaconda` to manage your Python packages. See the [conda installation instructions](https://docs.anaconda.com/anaconda/install/) and make sure you have conda up and running. Next:

- Update conda after the installation to make sure your version is up-to-date,
     ```
     conda update conda
     ```

- Make an environment file `environment.yml` with the dependencies of your own project.

     ```
     name: MY_ENVIRONMENT
     channels:
     - defaults
     - conda-forge
     dependencies:
     - python=3.10
     - ...
     - ...
     ```

- Setup/update the `environment`: Dependencies are collected in the conda `environment.yml` file (inside the root folder), so anybody can recreate the required environment using,

     ```
     conda env create -f environment.yml
     conda activate MY_ENVIRONMENT
     ```
     or alternatively, to update the environment (needed after adding a dependency),
     ```
     conda activate MY_ENVIRONMENT
     conda env update -f environment.yml --prune
     ```
     
     When creating or upating an environment, *solving the environment* can take very long if you have a lot of dependencies, so be wary of adding unnecessary dependencies.

- Install the `pySODM` code inside the environment of your own project (in `-e` edit mode):

     ```
     conda activate MY_ENVIRONMENT
     pip install -e .
     ```

     __Note:__ This step needs to be done in a terminal or command prompt. Use your favorite terminal or use the [Anaconda Prompt](https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-anaconda-prompt). Navigate with the `cd` command to the directory where you copied the `pySODM` repository.


_Optional_: To use the code, the general installation instruction outlined above suffice. When you're planning to work on the documentation or the code of the model implementations itself, make sure to also install the development requirements:

```
conda activate MY_ENVIRONMENT
pip install -e ".[develop]"
```
