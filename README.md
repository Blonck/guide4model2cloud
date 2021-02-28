# Idea

This is an exemplary repository for a project setup to create, maintain, and
deploy a machine learning model as a web API on a cloud provider. Of course,
this guide/repository contains all the steps necessary to build a machine
learning model. But it will also include all the small stuff needed to run it
as a project in the real world, which includes: working on it as a team,
setting up data pipelines to cope with new data, and providing the results as a
resource for other developers. Sayed that, I must admit that the most
complicated part is left out: how to get a dataset containing the labels in the
first place.

For that, I use the freely available data from a marketing campaign
[https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#]. Predicting a
marketing campaign's success may sound not very interesting, and honestly, it
is not. However, the data set is suited for this task for multiple
reasons. Most importantly, it is freely available. It has medium-size, which
means handling the data and training models, doesn't need an enormous amount of
computing resources. Also, the problem is not too easy to solve, as the
dataset is quite imbalanced.


The complete guide contains:
1. Setting up the environment
2. Rapid start to a working prototype
7. (TODO) Creating a docker image for the API
3. (TODO) Exploring the data
5. (TODO) Setting up data pipelines via DVC [https://dvc.org/]
6. (TODO) Developing an API for the model via fastapi [https://fastapi.tiangolo.com/]
8. (TODO) Trying different models and tune hyperparameters
9. (TODO) Deploying the model to
    * AWS
    * Google cloud
    * Microsoft Azure


# Setting up the environment

## Conda

For the package installation and handling, the conda package manager is used.
You can download the installer from [https://docs.conda.io/en/latest/miniconda.html#linux-installers] and install it via
```
bash Miniconda3-latest-Linux-x86_64.sh
```

Since conda sometimes needs some time to resolve package dependencies, the
first package I install is mamba, a drop-in replacement for conda. However, be
warned, in its current state, mamba is experimental.
```
conda install mamba -n base -c conda-forge
```

The whole development is now done inside a conda environment, which can be
directly created from the XML file in this repo:
```
mamba env create --file model2cloud.yml
```
Now the environment is activated via:
```
conda activate model2cloud
```

New packages can be installed by
```
mamba install $packagename
```
or better by adding it to the yml file followed by
```
mamba env update --file model2cloud.yml
```

## Jupytext - reviewable code

Jupytext is a tool to convert jupyter notebooks (ipynb) to plain python files,
see [https://jupytext.readthedocs.io/en/latest/].

Jupytext has three main use ases: applying linting to notebooks, editing
notebook files with your favorite editors, and doing version control and enable
code review on your notebooks. For this repo, the third point is the important
one. Especially, in teams you might want to use code review before merging new
code into the master.

One can manually convert a notebook to a python file
via:
```
jupytext --to py $notebook_file
```
However, better is to enable the synchronization between the notebook via the jupyter UI.
For this project, the synchronization is enabled for notebooks the following line
```
default_jupytext_formats = "ipynb,py:percent"
```
in the `jupytext.toml` file.

This means, whenever you save your notebook, a plain python file is created or updated.

### Jupytext commit hook

In principle, one could just commit the python to the git repository. There are
much better suited for reviewing the code and creating comments. When somebody
checks out the python file, the corresponding notebook is automatically
created. However, to understand content of a notebook the generated images are
quite crucial. As the ideas and findings presented in a notebook, are probably
more important than the code itself, I find it important to have all the
generated output available in a review.



## Jupyter notebooks

# Rapid start to a working prototype

## Fist look into the data

## First stupid model

## First fastapi application

## Make API available via docker
