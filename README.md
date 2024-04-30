# Microslope
is opiniated [micrograd](https://github.com/karpathy/micrograd)

## Quickstart
```
pyenv virtualenv 3.11.1 environ
pyenv activate environ
cd /path/to/microslope
sh setup.sh
```

Set the `PYTHONPATH` and go to the `src` folder
```
export PYTHONPATH="/path/to/microslope/src"
cd /path/to/microslope/src
```

To make sure that microslope works
```
pytest
```

You can access the draw function in the notebooks
```
cd path/to/microslope/notebooks
jupyter notebook
```

