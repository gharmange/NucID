# NucID
The purpose of this package is to rapidly identify the locations of nuclei in images. NucID uses a neural networks to identify nuclei 
(specifically the yolov7 architecture). This repository contains pre-trained weights to easily identify nuclei in your images, but if these weights do not
perform well on your images this repository also contains scripts to easily train the neural net on your cell type without needing to do any hand labeling of data.
Below is a step by step guide on how to use NucID.

## Setting up for use on Google Colab (you should only need to do this once)
Google Colab gives access to GPUs which makes running NucID significantly faster. An issue with google Colab is that every time you open a session
it does not save any previously installed packages. To avoid the need to install large packages such as deepcell each time we will create a folder
on google drive containing the necessary packages.

1. The first step is to download NucID from GitHub by clicking the green "Code" button above and selecting "Download ZIP." You should now have a file in your download folder called "NucID-main". Rename this folder to "NucID".

2. Go to your google drive account and, in the "My Drive" folder, create a folder called "Colab" and add to it the NucID folder downloaded from this repository.

3. You should now be able to open "RunNucID.pyb" if you want to run nuclei detection on images, or open "TrainNucID.pyb" if you want to train the model on new images
simply by clicking on the file. If when you click on the file it does not open, follow the prompts to download google Colab. 

## Identifying Nuclei in Images
1. Go to the NucID folder in your google drive through the browser and double click on it. This should open the notebook in Colab.

2. First go to the Runtime tab in the menu at the top left of the screen and select the "Change runtime type" option. If it is not already, make sure you select GPU as your Hardware

3. Run the first cell in the notebook labeled "Connect to Google Drive and install packages". Running this cell will cause a few prompts to show up on your screen, follow them to connect your google drive account containing the NucID folder. It will also download a couple pacakges needed to run the code. This section of code needs to be run once everytime you open the notebook.

4. Next start filling out the variable in the "Fill out variables and load packages". Here is an overview of each parameter:
   
   - **TIF_PATHS:** here we use glob to get a list of all image paths (images need to be in tif format) you want to run NucID on.
   - **RunPixelSize:** This value is the um per pixel of the images you are inputing. If you are using Shaffer Lab scope with 2x2 binning these are the
um/pixel values to use for each objective: 4X=3.25, 10X=1.29, 20X=.65, 60X=.22 

   - **NucleusChannel:** This is the channel of your image that contains the nuclei you want to identify (uses base 1).
   - **WeightsPath:** This is the path to the weights the nueral network uses to identify nuclei. The weights pubslished with this repository can be found in the directory "NucID/weight/". For the best results most of the time I suggest you use the "10X_ms.pt" weight file in that directory.
   - **PackagePath** This is the path to all the packages in the NucID repository. Simply enter the path to the "packages" folder in the NucID folder.
   You should never need to change the rest of the paramaters with regular runs of NucID, but here are the explanations if you are going beyond base use:
   - **tileSize** This sets the final tile size that is input into the neural network. The Yolov7 model expects a 640X640 tile, providing a different tile size will cause the model to reshape the input image to 640X640
   - **TrainPixelSize** This is the um/pixel size of the images the model was trained on. This parameter along with the RunPixelSize ensures that images are scaled properly.
   - **overlap** This parameter sets how much overlap is between tiles. overlap is necessary for stitching together locations of each cells in the large image. with a 640*640 a .1 value (10% overlap) works well.
