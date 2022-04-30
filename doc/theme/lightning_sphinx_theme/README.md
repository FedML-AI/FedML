# PyTorch-Lightning Sphinx Theme

Sphinx theme for [PyTorch-Lightning Docs](https://pytorch-lightning.readthedocs.io/en/latest/) based on the [Read the Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/en/latest).

## Setup the project for local development
This theme requires running both python commands and javascript (npm) commands.

### Step 0: Make sure you're on the conda environment you are using for pytorch-lightning
```bash
conda activate my-pl-env
```

### Step 1: Python setup
First, install all the docs deps for lightning
```bash
cd /path/to/pytorch-lightning

# install the docs requirements
git submodule update --init --recursive
pip install -r requirements/docs.txt
```

Setup the lightning_sphinx_theme
```
cd /path/to/lightning_sphinx_theme

# install project
python setup.py install

# install deps
pip install -r docs/requirements.txt
```

If you're on a mac with conda, and you get this error:
```
>> Pandoc wasn't found.
>> Please check that pandoc is installed:
>> https://pandoc.org/installing.html
>> Exited with code: 2.
```

then try the following command ([from this answer](https://stackoverflow.com/questions/62398231/building-docs-fails-due-to-missing-pandoc))
```
pip uninstall pandoc
conda install pandoc
```

### Step 2: setup the javascript things 🤮 
First, install the things in package.json
```bash
# run yarn install (uses `package.json`)
# you need node version 8.4.0
yarn install
```

[Install NPM](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) then run:

```
npm install
```

For good measure, make sure your npm and ruby paths are in your .bashrc or .zshrc 
```bash
# ruby
export PATH="/usr/local/opt/ruby/bin:$PATH"

# add npm
export PATH=/usr/local/share/npm/bin:$PATH
```

Make sure `grunt` works (we use grunt to see changes in real-time... ie: `hot-reload`)
```bash
grunt
```

Install a few npm packages
```bash
# provides hot reload for dev
sudo npm install -g grunt-cli 

# ?
sudo npm install -g sass
```

## Link the theme to the Lightning docs
Create an `.env.json` file that connects this theme to the lightning docs.

```bash
cd path/to/lightning_sphinx_theme
touch .env.json
```

Now copy paste the following into the `.env.json`
```json
{
    "DOCS_DIR": "path/to/pytorch_lightning/docs/"
}
```

## Development
Run the docs this way

```
grunt --project=docs
```

Building this will be slow at first... we recommend you disable the notebooks building (temporarily) to vastly speed up your docs development speed. To do this:
```bash
cd /path/to/pytorch-lightning/docs
ls
# (you'll see the conf.py file here). edit this document
```

In the conf.py file enable this flag
```bash
# default is false
_FAST_DOCS_DEV = False

# to build fast (not building the notebooks)
_FAST_DOCS_DEV = True
```


### Optional: build the demo docs
The lightning_sphinx_theme repo has a "demo" project (not lightning docs) that show you the styles very quickly.

First add the following entry to `.env.json`. 

```json
{
    "DOCS_DIR": "path/to/pytorch_lightning/docs/",
    "TUTORIALS_DIR": "path/to/tutorial/directory/docs"
}
```

and now build the "demo" docs with this command

```
grunt --project=tutorials
```

The resulting site is the lightning docs with the ability to change the styles.

## Testing your changes and submitting a PR

When you are ready to submit a PR with your changes you can first test that your changes have been applied correctly against either the PyTorch Docs or Tutorials repo:

1. Run the `grunt build` task on your branch and commit the build to Github.
2. In your local docs or tutorials repo, remove any existing `pt_lightning_sphinx_theme` packages in the `src` folder (there should be a `pip-delete-this-directory.txt` file there)
3. In `requirements.txt` replace the existing git link with a link pointing to your commit or branch, e.g. `-e git+git://github.com/{ your repo }/lightning_sphinx_theme.git@{ your commit hash }#egg=pt_lightning_sphinx_theme`
4. Install the requirements `pip install -r requirements.txt`
5. Remove the current build. In the docs this is `make clean`, tutorials is `make clean-cache`
6. Build the static site. In the docs this is `make html`, tutorials is `make html-noplot`
7. Open the site and look around. In the docs open `docs/build/html/index.html`, in the tutorials open `_build/html.index.html`

If your changes have been applied successfully, remove the build commit from your branch and submit your PR.

## Publishing the theme

Before the new changes are visible in the theme the maintainer will need to run the build process:

```
grunt build
```

Once that is successful commit the change to Github.

### Developing locally against PyTorch Docs and Tutorials

To be able to modify and preview the theme locally against the PyTorch Lightning Docs and/or the PyTorch Lightning Tutorials first clone the repositories:

- [PyTorch Lightning (Docs)](https://github.com/pytorch/pytorch)
- [PyTorch Lightning Tutorials](https://github.com/pytorch/tutorials)

Then follow the instructions in each repository to make the docs.

Once the docs have been successfully generated you should be able to run the following to create an html build.

#### Docs

```
# in ./docs
make html
```

#### Tutorials

```
# root directory
make html
```

Once these are successful, navigate to the `conf.py` file in each project. In the Docs these are at `./docs/source`. The Tutorials one can be found in the root directory.

In `conf.py` change the html theme to `pt_lightning_sphinx_theme` and point the html theme path to this repo's local folder, which will end up looking something like:

```
html_theme = 'pt_lightning_sphinx_theme'
html_theme_path = ["../../../lightning_sphinx_theme"]
```

Next create a file `.env.json` in the root of the THEME repo with some keys/values referencing the local folders of the Docs and Tutorials repos:

```
{
  "TUTORIALS_DIR": "../tutorials",
  "DOCS_DIR": "../pytorch_lightning/docs/source"
}

```

You can then build the Docs or Tutorials by running

```
grunt --project=docs
```
or

```
grunt --project=tutorials
```

These will generate a live-reloaded local build for the respective projects available at `localhost:1919`.

Note that while live reloading works these two projects are hefty and will take a few seconds to build and reload, especially the Docs.

### Built-in Stylesheets and Fonts

There are a couple of stylesheets and fonts inside the Docs and Tutorials repos themselves meant to override the existing theme. To ensure the most accurate styles we should comment out those files until the maintainers of those repos remove them:

#### Docs

```
# ./docs/source/conf.py

html_context = {
    # 'css_files': [
    #     'https://fonts.googleapis.com/css?family=Lato',
    #     '_static/css/pytorch_theme.css'
    # ],
}
```

#### Tutorials

```
# ./conf.py

# app.add_stylesheet('css/pytorch_theme.css')
# app.add_stylesheet('https://fonts.googleapis.com/css?family=Lato')
```

### Top/Mobile Navigation

The top navigation and mobile menu expect an "active" state for one of the menu items. To ensure that either "Docs" or "Tutorials" is marked as active, set the following config value in the respective `conf.py`, where `{project}` is either `"docs"` or `"tutorials"`.

```
html_theme_options = {
  ...
  'pytorch_project': {project}
  ...
}
```
