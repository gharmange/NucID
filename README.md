# NucID
The purpose of this package is to rapidly identify the locations of nuclei in images. NucID uses a neural networks to identify nuclei. This repository contains pre-trained weights to easily identify nuclei in your images, but if these weights do not
perform well on your images this repository also contains scripts to easily train the neural net on your cell type without needing to do any hand labeling of data. Below is a step by step guide on how to use NucID.
NOTE: This package mostly consists of functions that help you use yolov7 for nuclei. You can find the yolov7 repository here: https://github.com/WongKinYiu/yolov7

<h3 align="center">Example output of NucID</h3>
<p align="center">
  <img src="https://github.com/gharmange/NucID/blob/main/images/NucID_example.png" width="500" height="500">
</p>

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
   -  **Confidence_thresh** This is the threshold of confidence the model has to have that somethign is a nuclei to save it. If you have a threshold you know works just set it here and your ouputs will be directly what you want, however, if you are not sure of the confidence threshold to use you can set the threshold to 0 so that it saves everythign the model thinks may be a nucleus, and then later filter trying a few confidence thresholds using the "FilterCoords" function described in step 7 below.
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
   
## Training NucID
To train NucID you need an image of nuclei in the tif format, along with a corresponding mask file in the tif format where each mask has a unique pixel value. To get this data you can use online repositories such as the one from deepcell https://datasets.deepcell.org (the pre-trained weights published with this package was partially trained on these), or you can generate your own masks. There are ways to generate your own masks by anotating by hand but I have found that existing deep learning cell segmenters such as deepcell (found here: https://github.com/vanvalenlab/deepcell-tf) or cellpose (found here:https://github.com/MouseLand/cellpose) usually do a good job generating the masks for you. Since these models do a good job finding and segmenting nuclei, why used NucID? Both these segmenting models are quite slow and resource intensive, infact NucID is more than 10X faster than deepcell and uses a small fraction of the resources. Therefor, we can use these more resourse and time intensive models to generate a training data set and then only use the faster NucID model for frequent use. Below is a step by step guide on how to train NucID with your own anotated data.

NOTE: Unlike running NucID both generating training data and training the model uses more resources than a free colab session provides. To run training you will need to pay for colab pro, or run on a computer with the appropriate resources (you need a minimun of ~20GB of RAM and ~14GB of VRAM).

1. Get images of nuclei the current model does not do a good job of identifying and run them through a pipeline to generate masks. I have made a package that makes it easy to use deepcell and cellpose to generate masks. Go to this link and follow the intructions : https://github.com/SydShafferLab/DeepCellHelper

2. Once you have generated masks for your images you need to generate 640X640 images of nuclei and a correspoinding file with bouding boxes for each nuclei. To start doing this open up the "TrainNucID.ipynb" notebook which is located in the NucID repository and run the first cell to connect to google drive, and stall pacakages, and import packages (if you need more guidance with this step see the first few steps of the "Identifying Nuclei in Images" section above).

3. The first thing to do is to check that your masks are doing a good job identifying real nuclei and not selecting noise (use this package to check masks: https://github.com/gharmange/CellQuant). Sometimes very small objects and/or very large objects that are clealy not nuclei are being called as nuclei. In that case you can try filtering the masks with simple size thresholds using the function `FilterMask`, skip this step if your masks look good. This function takes in a mask file path, a minimum and a mximum mask area in pixels, and outputs a filtered mask file. If you want to check the results of the filtering you can set the check_mask variable to True and provide the path to the image file that corresponds to the mask file. An example implementaion of this is in the cell labeled "Filter Mask Files". 

4. Once you are happy with your masks we can move on to generating the 640X640 image tiles with their corresponding bounding box files, this is done with the `MakeTrainingData` function. this function takes in an image path, its coresponding mask path, a path for the outputs (which must exist already), and the channel the nuclei images are in. It will output a training data folder with an images folder containing the images and a labels forlder containing the correspoinding bounding box information. This fuction is implamented along with the making of the validation data set in the section labeled 'Generate Training Data'. The validation data is generated using the `MakeValData` function and takes in the path to the training data as well as the number of validation images you wants and outputs this data into a 'val' folder containing the validation images and labels. NOTE: if you wan to balance you data set in the next step DO NOT generate the validation data set here. To run these functions fill out the varibales in the first cells of the 'Generate Training Data' section, run it, and then run the second cell.

5. Once you have generated the training data it is a good idea to check that the conversion of images and mask to 640X640 tiles and bounding boxes has worked properly. To do this you can run the 'CheckBB' function. This function takes in the path to an image an the path to its corresponding label file and plots an image showing the bounding boxes around the nuclei. If the bouding boxes do not look right you will need to trouble shoot the steps above to figure out what went wrong. The ChekBB function is implemented in the section labeled 'Check Training Data'.

example output of `CheckBB`

6. Nuclei can be very dense or sparse, if your training data has a very uneven number of very dense or very sparse nuclei the model may become bias to either over call or under call the number of nuclei in an image. To avoid this bias we can try and balance the data using the `FindBalanceData` and `BalanceData` funtions. The `FindBalaceData` takes as input the directory to the labels of your training data, how many bins you want to split this data in, and how many images to take per bin. This function determins how many nuclei ther are in each image, bind these images depending on how many bins you specify and then takes the number of files you specified per bin. The output of the `FindBalaceData` is a list of paths to the labels it thinks will make a balanced data set. If you are happy with the number of files that will end up in the balanced data set then you can input this list into the `BalanceData` wich will copy the labels and their corresponding images into a new Balanced folder. An example running these functions can be found in the section labeled "Balance Training Data".

7. Once the Balanced data set is establised you can run the `MakeValData` function inputing the path to the balanced folder and the number of files you wan in your validation data set and the funtion will output a "val" folder with the images and the labels. An example of this can be found in the last cell of the section labeled "Balance Training Data".

8. Now that we have a training and validation data set we need to fill out a file to tell the model where to look for the data. In the NucID folder go the data folder in the yolovy package it will be at this directory: '/NucID/packages/yolov7/data'. in this direcory open up the file called 'Nuclei.yaml' and edit the path next to 'train:' and 'val:' so that they point to your newly generated training and validation data. Save the file and close out of it.

9. You also have the option to change hyperparmeters for training the model. I do not do this and always run with the defaults bu the file containing the parameters can be found here: '/NucID/packages/yolov7/data/hyp.scratch.custom.yaml'

10. We are now ready to input the command to train the model. First for the model to train you need your current direcotyr to be the yolov7 package folder (should be this directory: '/NucID/packages/yolov7'). The command for training is run in terminal (or in the notebook using the correct escape character) and then running `python training.py` followed by the following parameters:

**-- device** Device tell the model if model is running on GPU (enter the gpu number ex. 0) or CPU enter ('cpu'). since training the model would take way too long on a cpu you will almost always enter `-- device 0`

   
            

