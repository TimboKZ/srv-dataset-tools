# Texture reconstruction data set tools

This repo contains the tools for texture reconstruction based on laparoscopic surgery data sets, made for Surgical 
Vision and Robotics group at UCL. This repository goes together with Timur Kuzhagaliyev's final year project, 
described in [this report](#) (link coming soon).


# Project structure

The purpose of each directory is described below:

* `ds_tools/`: This folder contains all of the Python source code for the project. Note that it's meant to act a 
self-contained Python package, meaning that code inside it uses package-absolute imports, e.g. 
`from ds_tools.shared import util`.
* `notebooks/`: This folder contains various Jupyter notebooks, mostly used for calibration purposes. The reason the 
calibration code is presented in the form of Jupyter notebooks is to make it easier to select the correct filters and
 debug the calibration process. Notes on how to use the notebooks can be found below.
* `data/`: The folder meant to hold data used for texture reconstruction. Since this usually includes large video and
 CSV files, the contents of this folder are ignored via `.gitignore`. 

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

You are free to adjust the paths and filenames for your convenience. Note that you don't *have* to put your data into 
the `data/` folder - you can use your own absolute paths, e.g. `/home/my_user/my_data/pose_ecm.csv`. Keep in mind 
that the **output** some scripts generate might go into the `data/` folder by default.


# Using Jupyter notebooks

The `notebooks/` folder and its sub-directories contain Jupyter notebooks that showcase various tools from this 
repository. The Python import statements in these notebooks are relative to the `.ipynb` files, which means that the 
notebooks should work regardless of where you run `jupyter notebook`. That said, it's recommended to start the 
Jupyter server as follows:
```bash
cd notebooks/
jupyter notebook
```

