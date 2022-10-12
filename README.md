# NucID
The purpose of this package is to rapidly identify the locations of nuclei in images. NucID uses a neural networks to identify nuclei 
(specifically the yolov7 architecture). This repository contains pre-trained weights to easily identify nuclei in your images, but if these weights do not
perform well on your images this repository also contains scripts to easily train the neural net on your cell type without needing to do any hand labeling of data.
Below is a step by step guide on how to use NucID.

## Setting up for use on Google Colab (you should only need to do this once)
Google Colab gives access to GPUs which makes running NucID significantly faster. An issue with google Colab is that every time you open a session
it does not save any previously installed packages. To avoid the need to install large packages such as deepcell each time we will create a folder
on google drive containing the necessary package and then mount google drive to google Colab.

1. The first step is to download NucID from GitHub by clicking the green "Code" button above and selecting "Download ZIP." You should now have a file in your download folder called "NucID-main". Rename this folder to "NucID".

2. Go to your google drive account and, in the "My Drive" folder, create a folder called "Colab" and add to it the NucID folder downloaded from this repository.

3. You should now be able to open "RunNucID.pyb" if you want to run nuclei detection on images, or open "TrainNucID.pyb" if you want to train the model on new images
simply by clicking on the file. If when you click on the file it does not open, follow the prompts to download google Colab. 

## Identifying Nuclei in Images
