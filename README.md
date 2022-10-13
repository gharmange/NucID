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

3. You should now be able to open "RunNucID.ipynb" if you want to run nuclei detection on images, or open "TrainNucID.ipynb" if you want to train the model on new images
simply by clicking on the file. If when you click on the file it does not open, follow the prompts to download google Colab. 

## Identifying Nuclei in Images
1. Go to the NucID folder in your google drive through the browser and double click on it. This should open the notebook in Colab.

2. First go to the Runtime tab in the menu at the top left of the screen and select the "Change runtime type" option. If it is not already, make sure you select GPU as your Hardware

3. Run the first cell in the notebook labeled "Connect to Google Drive and install packages". Running this cell will cause a few prompts to show up on your screen, follow them to connect your google drive account containing the NucID folder. It will also download a couple pacakges needed to run the code. This section of code needs to be run once everytime you open the notebook.

4. Next fill out the variable in the "Fill out variables and load packages" and run it. Here is an overview of each parameter:
   
   - **TIF_PATHS:** here we use glob to get a list of all image paths (images need to be in tif format) you want to run NucID on.
   - **RunPixelSize:** This value is the um per pixel of the images you are inputing. If you are using Shaffer Lab scope with 2x2 binning these are the
um/pixel values to use for each objective: 4X=3.25, 10X=1.29, 20X=.65, 60X=.22 

   - **NucleusChannel:** This is the channel of your image that contains the nuclei you want to identify (uses base 1).
   - **WeightsPath:** This is the path to the weights the nueral network uses to identify nuclei. The weights pubslished with this repository can be found in the directory "NucID/weight/". For the best results most of the time I suggest you use the "10X_ms.pt" weight file in that directory.
   - **PackagePath** This is the path to all the packages in the NucID repository. Simply enter the path to the "packages" folder in the NucID folder.
   
   You should never need to change the rest of the paramaters with regular runs of NucID, but here are the explanations if you are going beyond base use:
   - **tileSize** This sets the final tile size that is input into the neural network. The Yolov7 model expects a 640X640 tile, providing a different tile size will cause the model to reshape the input image to 640X640
   - **overlap** This parameter sets how much overlap is between tiles. overlap is necessary for stitching together locations of each cells in the large image. with a 640X640 a .1 value (10% overlap) works well.
   - **TrainPixelSize** This is the um/pixel size of the images the model was trained on. This parameter along with the RunPixelSize ensures that images are scaled properly.
 
5. Next we initialize a NucID object by running the cell labeled "Initate NucID object". What this does is it load all the variables into an object, in this case named "nid" but you can assign any name to this object and generate multiple instances of this object. Now that this object is initiated you can run specified funtios on it such as "RunNucID" by entering `nid.RunNucID('tif/path')` (we do this in the next cell). This object will also contain useful
information you can access. For example you can access paths of where things are saved such as the path to the tif that was run by running `nid.tif_path`, or get the array of the image that was just run by entering `nid.image`.

6. Once the object has been initiated we can run different funtions with this object. Here is a list of funtions and what they do:
   - **RunNucID** This funtion takes as input the path to a tif file and ouputs a csv file in the same location as the tif file contain the x and y coordinates of each cell in the image as well as a 3rd column saying how confident the model is that that point is a nucleus. An example on how to run this can be found in the cell labeled "Run NucID" in the RunNucID.ipynb notebook.
   - **TestModel** This funtion is used to quickly test if the model will work well in an image. It takes in a path to a tif file you wan to check, as well as an xy coordinate of where in that image you would like to test the model (the model will output the locations of nuclei in a 640X640 tile around this point). It will then plot an image of what the model is calling as a nucleus and with what confidence. An example on how to run this can be found in the cell labeled "Test model" in the RunNucID.ipynb notebook.
   
7. There are also other useful function in NucID to test and check the model that can be run without intializing a NucID object.
   - **checkNucXY** This fuction takes the ouput of the RunNucID funtion and overlays on an image what it is calling a nucleus and with what confidence. This funtion take as input the path to the tif_file of interest and its respectice coordinate file. An example on how to run this can be found in the cell labeled "Check if output coodinates are correct" in the RunNucID.ipynb notebook.
   - **FilterCoords** This funtion is used to filter out neclei detected under a certain confidence threshold. When running NucID you have the option of what confidence threshold nuclei have to meet to keep them if this threshold was set to low you can generate a new coordinate set with this function with a highter threshold. In fact I recomend you run NucID with a confidence threshold of 0 to get all possible nuclei and then filter after the fact so you never have to re-run the more time consuming nucleus identification process again. This funtion will take in the path to the coordinate file you want to filter and a confidence threshold, and will ouput a new coordinate file labeled with the confidence threshold used. An example on how to run this can be found in the cell labeled "Filter output coodinates by confidence" in the RunNucID.ipynb notebook.
   
            

