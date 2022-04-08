# Introduction

This is a tool for the course DBL Process Mining by group 22.
The tool consists of multiple jupyter notebooks and some python scripts.

Authors of the project:
- Blazej Nowak: 1617303
- Calvil Lee: 1623818
- Caspar Nijssen: 1604031
- Jarl Kusters: 1503863
- Rodigo de Miguel: 1571664
- Pim Oude Vekdhuis: 1624156

# Dependencies

The program uses the following python libraries, the version at which the tool was written will be displayed in brackets e.g. `Python` (== 3.9.11)):

- `Pandas`    (==1.4.1)
- `Numpy`     (==1.22.3)
- `sklearn`   (==1.0.2)
- `psutil`    (==5.8.0)

If these are not available or are not up to date, it is suggested to create an environment with these versions for optimal use.

# How to run

1. Open the `config.ipynb` file in either a `text editor`, jupyter notebook or preferred equivalent. Here fill in the path were the data is stored and specify the train and test set as instructed in the config.
   During our project we opted to use absolute paths.
2. As the `main.ipyn`b is used to run all dependencies and models, after properly using the configuration, pressing run all will run all the models, giving back a print of all the error statistics and run times, 
   while sperate csv files will be created for the outcomes.
