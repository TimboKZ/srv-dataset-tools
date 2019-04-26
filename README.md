# Texture reconstruction data set tools

This repo contains the tools for texture reconstruction based on laparoscopic surgery data sets, made for Surgical 
Vision and Robotics group at UCL. This repository goes together with Timur Kuzhagaliyev's final year project, 
described in [this report](#) (link coming soon).


# Project structure

The purpose of each directory is described below:

* `ds_tools/`: This folder contains all of the Python source code for the project. Note that it's meant to act a 
self-contained Python package, meaning that code inside it uses package-absolute imports, e.g. 
`from ds_tools.shared import util`.
* `matlab_issi/`: Matlab code used to process data from the da Vinci surgical robot (acquired using a dvLogger). It's
 mostly based on the ISSI file processing functions provided by Intuitive Surgical Inc.
* `notebooks/`: This folder contains various Jupyter notebooks, mostly used for calibration purposes. The reason the 
calibration code is presented in the form of Jupyter notebooks is to make it easier to select the correct filters and
 debug the calibration process. Notes on how to use the notebooks can be found below.
* `data/`: The folder meant to hold data used for texture reconstruction. Since this usually includes large video and
 CSV files, the contents of this folder are ignored via `.gitignore`. 

# Python packages

This project uses Python 3. You will need the following Python Pip packages to run the code in this project:
```
opencv-python
numpy
pandas
```

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

# User manual

This section provides instructions on how to use each individual part of the project.

## Converting dvLogger output CSV files

dvLogger recorder is the tool used to record da Vinci kinematics and stereo endoscope footage. It starts logging data
 the moment it boots up, given it's connected to the da Vinci surgical system. The data for each recording session 
 contains multiple `.issi` files and 2 `.avi`. The two videos are the video footage from the left and right stereo 
 endoscope cameras respectively, and 3 of the `.issi` contain da Vinci kinematics and timestamps for video frames for
 each camera. Typical output of the dvLogger looks like this:
 ```
.
├── ...
├── DaVinciSiMemory.issi
├── EndoscopeImageMemory_0.avi
├── EndoscopeImageMemory_0.issi
├── EndoscopeImageMemory_1.avi
└── EndoscopeImageMemory_1.issi
```
  
To make these `.issi` files easier to use with Python, they must first be converted into the `.csv` format. This can 
be done using the `ConvertAll.m` script from the `matlab_issi/` folder. Place the `.issi` files listed above into 
`matlab_issi/`, then either run `ConvertAll.m` or run the following Matlab code directly:
```matlab
dvsm = ISSILoad('DaVinciSiMemory.issi');
eim0 = ISSILoad('EndoscopeImageMemory_0.issi');
eim1 = ISSILoad('EndoscopeImageMemory_1.issi');

dvsmData2csv(dvsm, 'DaVinciSiMemory.csv');
eimData2csv(eim0, 'EndoscopeImageMemory_0.csv');
eimData2csv(eim1, 'EndoscopeImageMemory_1.csv');
```

This code will convert all 3 files into CSV format, and save them into `matlab_issi/`, but it might be a good idea to
 move them to the `data/` folder. Note that this Matlab code can only be ran on a Windows edition of Matlab, since 
 some of the proprietary ISSI loading code from `matlab_issi/` folders uses encrypted binaries which are only 
 compatible with Windows.
 
## Syncing timestamps in dvLogger data
 
The timestamps in the CSV files generated above are not synced by default - some of the might skip several 
frames or have slightly different rates. The 3 CSV files and 2 AVI files need to be processed further to make sure 
the data is in sync. This can be done using the `sync_timestamps.py` script from the `ds_tools/scripts/` folder. Open
`sync_timestamps.py` and change the `capture_dir` variable to reflect the directory where your files are located.
 For example, if you were to place the files in `data/my_capture/`, you will have to update `capture_dir` to look as 
 follows:
```python
# ...
def main():
    data_dir = util.get_data_dir()
    capture_dir = path.join(data_dir, 'capture_dir')
    # ...
```

You can then run the script using:
```bash
python3 ds_tools/scripts/sync_timestamps.py
```

Once the script terminates, a folder called `synced/` will be created inside the directory you specified. This folder
 will contain the processed CSV files and AVI videos, all synced to 30 FPS.
 
## Segmenting 3D models from CT scans, UV unwrapping

CT scans almost always come as a collection of DICOM images, where each image represents a single slice through the 
volume of the scanned scene. `.DICOM` files can be viewed using [itk-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php).
 Relevant objects, such as markers on the calibration object or markers on the endoscope collar, can be easily 
 segmented from the CT scan using one of [itk-SNAP's automatic segmentation tools](http://www.itksnap
 .org/docs/viewtutorial.php?chapter=TutorialSectionIntroductionToAutomatic) (snakes evolution tool is recommended). 

Note that groups of markers, e.g. all endoscope collar markers, must all be segmented into the same label, and each 
 distinct group should have its own label. The recorded surface, e.g. the walls of the recorded surgical phantom, can
 be segmented using the same approach.
  
Once segmentation is done, each label should be exported as a separate `.stl` mesh. This can be done using a built-in
 itk-SNAP tool for mesh export.
 
The object of the mesh whose texture will be reconstructed (e.g. surgical phantom) has to go through extra processing
 steps, such as smoothing and UV unwrapping. These steps can be carried out using [Blender](https://www.blender.org/)
 software. The relevant `.stl` can be imported into Blender using the built-in STL plugin (with orientation Z-up 
 Y-forward and same scaling as original model).