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
2. Quickly to a functional prototype
    1. Simple RandomForesetClassifier
    2. FastAPI webapp [https://fastapi.tiangolo.com/]
    3. Docker image
3. (TODO) Setting up data pipelines via DVC [https://dvc.org/]
4. (TODO) Exploring the data
6. (TODO) Trying different models and tune hyperparameters via optuna [https://optuna.org/]
7. (TODO) Deploying the model to
    1. AWS
    2. Google cloud
    3. Microsoft Azure


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

In principle, one could commit the python to the git repository. There are
much better suited for reviewing the code and creating comments. When somebody
checks out the python file, the corresponding notebook is automatically
created. However, to understand the content of a notebook, the generated images are
quite crucial. As the ideas and findings presented in a notebook are probably
more important than the code itself, I find it essential to have all the
generated output available in a review.

This approach comes with two drawbacks. The git repository contains now all
images from the notebooks, increases in size significantly. If you want to
readable notebooks in the repository this is not avoidable. The second
drawback is that the ipynb and python files could come out of sync, this one
can prevent a pre-commit hook as given in `githooks/pre-commit`.

To enable these githooks you can configure your hook path via:
```
git config core.hooksPath githooks/
```

# Quickly to a functional prototype

So that is enough configuration for the beginning. The next part cares around
creating a prototype, including a preliminary model and exposing it via a web
API. The idea behind the preliminary model is to get as fast as possible to
running code which serves two purposes:
1. Give a baseline performance.
2. Provide other teams/developers something against which they can develop

Using the first model, one can already estimate how complex the final problem will be.
For example, if you need a prediction with 90% accuracy for production and the
preliminary model achieves already 85%, one could expect that this gap can be indeed
closed with a decent amount of work. If the preliminary model only reaches 60% accuracy,
it might be that you have a long way in front of you.

A working prototype also helps when cooperating with other teams or developers.
In the case, your prediction is used in a more complex application; others can already
develop their solution (frontend or backend application) against your model.
You can then improve your model iteratively, for example, in a Scrum or Kanban style.

## First look into the data

In general, one should strive for a deeper understanding of the data and it's
context, which allows to tailer the model to the use case and quench out the
last bit of performance.  However, the first model should be developed as fast
as possible. Still, a minimum amount of exploratory data analysis is needed to
generate any meaningful results.

In `notebooks/01_FirstLookIntoData.ipynb` I check at least who the target
variable looks like.  In particular, `data['y'].value_counts(normalize=True)`
shows that the dataset is imbalanced, 11% positive cases. This must be
respected even in the first prototype. In particular, the metric for the
model must be chosen carefully. Accuracy will be certainly not a very useful.

### NA's

Luckily, there are not NA values in this dataset (`data.isna().sum()`).
Otherwise, these have to be also handled for the prototype.

### Categorical variables

For the first model, I check that enough samples are present in the dataset for
each class in each category, and I don't need to group these together, which
only occur seldomly. 
![Count classes](./images/CountClasses.png)

## First stupid model

The random forest in `notebooks/02_FirstModel.ipynb` is trained on 80% of the
data and tested on the remaining 20%. To encode the categorical columns,
I use the OneHotEncoder from category_encoders [https://contrib.scikit-learn.org/category_encoders/].

For later experiments, the dataset must be split into a proper train,
validation, and test set, but having a single test set is ... acceptable for
the first model.

For now, I use balanced accuracy to evaluate the model. Also, this needs a
later revisit, but for now, the model achieves a balanced accuracy of 88%. Not
bad, for almost no work.  In the last section of the notebook, the model is
trained on the full dataset and saved as a pickle file for later usage.

In the last section of the notebook, the model is trained on the full dataset
and saved as a pickle file for later usage.

## First FastAPI application

The next step is to cast the ready model into an application. For that purpose, I use
the fantastic FastAPI library [https://fastapi.tiangolo.com/]

In `app/input.py`, I define the input data for the model. FastAPI automatically creates a schema from this using another
fantastic python library: pydantic [https://pydantic-docs.helpmanual.io/].

The actual app `app/simpleapp.py` mainly does two things.

Firstly it loads in the stored pickle files. The custom unpickler is needed since the
model is pickled within the notebook, and thus the context from which the app is executed
would not find the corresponding python classes.
```
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

encoder_file = 'model/simple_enc.pkl'
model_file = 'model/simple_rf.pkl'

logging.debug(f'Loading model from {model_file}')
model = CustomUnpickler(open(model_file, 'rb')).load()
encoder = CustomUnpickler(open(encoder_file, 'rb')).load()
```

The other relevant lines create the endpoint, transform the incoming data into a DataFrame,
execute the model, and return the prediction.
```
@app.post('/predict')
def predict(data: ClientDataSimple):
    # transform dict to pandas DataFrame
    data_as_dict = {key: [value] for key, value in data.dict().items()}
    df = pd.DataFrame.from_dict(data_as_dict)
    x = encoder.transform(df)
    preds = model.predict(x)
    preds = pd.Series(preds)

    # transform result back to original presentation
    preds = preds.map({0: 'no', 1: 'yes'})

    return preds.to_json(orient='records')
```

The application is now started via executed in the app directory.
```
uvicorn simpleapp:app --reload
```

That's it. Even better, FastAPI also provides also an OpenAPI specification under `http://127.0.0.1:8000/openapi.json`
and a Swagger UI under `http://127.0.0.1:8000/docs`, which one can use to test the API.

## Make API available via docker
