# 3D reconstruction data set tools

Tools for the creation of a 3D reconstruction data set for Surgical Vision and
Robotics group at UCL. This repository goes together with Timur Kuzhagaliyev's
Final Year Project, described in [this report](#) (link coming soon).


# Project structure

The purpose of each directory is described below:

* `ds_tools/`: This folder contains all of the Python source code for the project. Note that it's meant to act a 
self-contained Python package, meaning that code inside it uses package-absolute imports, e.g. 
`from ds_tools.shared import util`.

# Using custom data

As you might have noticed, this repository includes a `data/` folder. This folder is empty (except for a `.gitignore`
file) and is ignored by Git, which means that you can put anything in there and it won't affect your commits.

A lot of scripts and Jupyter notebooks in this repository assume that relevant files are located in the `data/` folder.
For example, the code that assumes that files `data/EndoscopeImageMemory_0.avi` and `data/pose_ecm.csv` exist will look
as follows:

```python
from os import path
from ds_tools.shared import util

data_dir = util.get_data_dir()
video_path = path.join(data_dir, 'EndoscopeImageMemory_0.avi')
pose_ecm_path = path.join(data_dir, 'pose_ecm.csv')
```

You are free to adjust the paths and filenames for your convenience. Note that you don't have to put your data into 
the `data/` folder - you can use your own absolute paths, e.g. `/home/my_user/my_data/pose_ecm.csv`. Keep in mind 
that the **output** some scripts generate might go into the `data/` folder by default.


# Using Jupyter notebooks
