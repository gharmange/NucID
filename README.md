# NucID
The purpose of this package is to rapidly identify the locations of nuclei in images. NucID uses a trained neural network to identify nuclei in a large variety of contexts. If the weights published in this repository do not
perform well on your images, this repository also contains scripts to easily train the neural net on your own images without needing to do any hand labeling of data. Below is a step-by-step guide on how to use NucID.
NOTE: This package mostly consists of functions that help you use yolov7 to identify nuclei. You can find the yolov7 repository here: https://github.com/WongKinYiu/yolov7

<h3 align="center">Example output of NucID</h3>
<p align="center">
  <img src="https://github.com/gharmange/NucID/blob/main/images/NucID_example.png" width="500" height="500">
</p>

## Setting up for use on Google Colab (you should only need to do this once)
Google Colab gives access to GPUs which makes running NucID significantly faster. An issue with google Colab is that every time you open a session
it does not save any previously installed packages. To avoid the need to install large packages such as yolov7 each time we will create a folder
on google drive containing the necessary packages.

1. The first step is to download NucID from GitHub by clicking the green "Code" button above and selecting "Download ZIP." You should now have a file in your download folder called "NucID-main". Rename this folder to "NucID".

2. Go to your google drive account and, in the "My Drive" folder, create a folder called "Colab" and add to it the NucID folder downloaded from this repository.

3. If you access google drive from the browser you should now be able to open "RunNucID.ipynb" if you want to run nuclei detection on images, or open "TrainNucID.ipynb" if you want to train the model on new images
simply by clicking on the file. If when you click on the file it does not open, follow the prompts to download google Colab.

## Identifying Nuclei in Images
1. Go to the NucID folder in your google drive through the browser and double-click on it. This should open the notebook in Colab.

2. First go to the Runtime tab in the menu at the top left of the screen and select the "Change runtime type" option. If it is not already, make sure you select GPU as your Hardware

3. Run the first cell in the notebook labeled "Connect to Google Drive and install packages". Running this cell will cause a few prompts to show up on your screen, follow them to connect your google drive account containing the NucID folder. Running this cell will also download a couple of packages needed to run the code. This section of code needs to be run once every time you open the notebook.

4. Next fill out the variable in the "Fill out variables and load packages" and run it. Here is an overview of each parameter:

   - **TIF_PATHS:** here we use glob to get a list of all image paths (images need to be in tif format) you want to run NucID on.
   - **RunPixelSize:** This value is the um/pixel of the images you are inputting. If you are using Shaffer Lab scope with 2x2 binning these are the
um/pixel values to use for each objective: 4X=3.25, 10X=1.29, 20X=.65, 60X=.22

   - **NucleusChannel:** This is the channel of your image that contains the nuclei you want to identify (uses base 1).
   - **WeightsPath:** This is the path to the weights the neural network uses to identify nuclei. The weights published with this repository can be found in the directory "/NucID/weight/". For the best results most of the time I suggest you use the "10X_ms.pt" weight file in that directory.
   -  **Confidence_thresh** This is the threshold of confidence the model has to have that something is a nucleus to save it. If you have a threshold you know works just set it here and your outputs will be directly what you want, however, if you are not sure of the confidence threshold to use you can set the threshold to 0 so that it saves everything the model thinks may be a nucleus. After saving everything you filter the nuclei by trying a few confidence thresholds using the "FilterCoords" function described in step 7 below.
   - **PackagePath** Enter the path to all the packages in the NucID repository. Specifically, enter the path to the "packages" folder in the NucID folder.

   You should never need to change the rest of the parameters with regular runs of NucID, but here are the explanations if you are going beyond base use:
   - **tileSize** This sets the final tile size that is input into the neural network. The yolov7 model expects a 640X640 tile, providing a different tile size will cause the model to reshape the input image to 640X640
   - **overlap** This parameter sets how much overlap is between tiles. overlap is necessary for stitching together nucleus locations in each tile into locations for the whole image. With a 640X640 tile the  .1 value (10% overlap) works well.
   - **TrainPixelSize** This is the um/pixel size of the images the model was trained on. This parameter along with the RunPixelSize ensures that images are scaled properly.

5. Next we initialize a NucID object by running the cell labeled "Initiate NucID object". What this does is it loads all the variables into an object, in this case named "nid" but you can assign any name to this object, and generate multiple instances of this object. Now that this object is initiated you can run specified functions on it such as "RunNucID" by entering `nid.RunNucID('tif/path')` (we do this in the next cell). This object will also contain useful
information you can access. For example, you can access paths of where things are saved such as the path to the tif that was run by running `nid.tif_path`, or get the array of the image that was just run by entering `nid.image`.

6. Once the object has been initiated we can run different functions with this object. Here is a list of functions and what they do:
   - **RunNucID** This function takes as input the path to a tif file and outputs a csv file in the same location containing the x and y coordinates of each cell in the image as well as a 3rd column with the confidence level the model has that that point is a nucleus. An example of how to run this can be found in the cell labeled "Run NucID" in the RunNucID.ipynb notebook.
   - **TestModel** This function is used to quickly test if the model will work well in an image. It takes in a path to a tif file you want to check, as well as the xy coordinate of where in that image you would like to test the model. It will then plot an image of what the model is calling as nuclei and with what confidence. An example of how to run this can be found in the cell labeled "Test model" in the RunNucID.ipynb notebook. The image at the top of this readme is an example of the output of this function.

7. There are also other useful functions in NucID to test and check the outputs of the model that are run without initializing a NucID object. Here is a list of these functions:
   - **checkNucXY** This function takes the output of RunNucID and overlays on an image what the model is calling a nucleus and with what confidence. This function takes as input the path to the tif_file of interest and its respective coordinate file. An example of how to run this can be found in the cell labeled "Check if output coordinates are correct" in the RunNucID.ipynb notebook.
   - **FilterCoords** This function is used to filter out nuclei detected under a certain confidence threshold. When running NucID you specify a confidence threshold nuclei have to pass to keep them in the dataset. If this threshold was set to low you can generate a new coordinate set with this function with a higher threshold. In fact, I recommend you run NucID with a confidence threshold of 0 to get all possible nuclei and then filter after the fact so you never have to re-run the more time-consuming nucleus identification process again. This function takes the path to the coordinate file you want to filter and a confidence threshold as input and will output a new coordinate file labeled with the confidence threshold used. An example of how to run this can be found in the cell labeled "Filter output coordinates by confidence" in the RunNucID.ipynb notebook.

## Training NucID
To train NucID you need an image of nuclei in the tif format, along with a corresponding mask file in the tif format where each mask has a unique pixel value. To get this data you can use online repositories such as the one from deepcell https://datasets.deepcell.org (the pre-trained weight file published with this package was partially trained on these), or you can generate your own masks. There are ways to generate your own masks by annotating by hand but I have found that existing deep learning cell segmenters such as deepcell (found here: https://github.com/vanvalenlab/deepcell-tf) or cellpose (found here: https://github.com/MouseLand/cellpose) usually do a good job generating the masks for you. Since these models do a good job of finding and segmenting nuclei, why used NucID? Both these segmenting models are quite slow and resource intensive, in fact, NucID is more than 10X faster than deepcell and uses a small fraction of the computational resources. Therefore, we can use these more resource and time-intensive models to generate a training data set and then only use the faster NucID model for frequent use. Below is a step-by-step guide on how to train NucID with your own annotated data.

NOTE: Unlike running NucID both generating training data and training the model uses more resources than a free colab session provides. To run training you will need to pay for colab pro or run on a computer with the appropriate resources (you need a minimum of ~20GB of RAM and ~14GB of VRAM).

1. Get images of nuclei the current model does not do a good job of identifying and run them through a pipeline to generate masks. I have made a package that makes it easy to use deepcell and cellpose to generate masks. Go to this link and follow the instructions: https://github.com/SydShafferLab/DeepCellHelper.

2. Once you have generated masks for your images you need to generate 640X640 images of nuclei and a corresponding file with bounding boxes for each nucleus. To start doing this open the "TrainNucID.ipynb" notebook which is located in the NucID repository and run the first cell to connect to google drive, install packages, and import packages (if you need more guidance with this step see the first few steps of the "Identifying Nuclei in Images" section above).

3. The first thing to do is to check that your masks are doing a good job identifying real nuclei and not segmenting noise (use this package to check masks: https://github.com/gharmange/CellQuant). If your masks look good skip this step, however, in some masks there are very small objects and/or very large objects that are clearly not nuclei that get assigned a mask. In this case, you can try filtering the masks with simple size thresholds using the function `FilterMask`. This function takes in a mask file path, a minimum, and a maximum mask area in pixels, and outputs a filtered mask file. If you want to check the results of the filtering you can set the check_mask variable to True and provide the path to the image file that corresponds to the mask file (note that if there are a lot of cells in this image this visualization can take time and use up a lot of RAM). An example implementation of this is in the cell labeled "Filter Mask Files".

4. Once the masks look good, we can move on to generating the 640X640 image tiles with their corresponding bounding box files. This is done with the `MakeTrainingData` function which takes in an image path, its corresponding mask path, a path for the outputs (which must exist already), and the channel the nuclei images are in (enter 1 if there is only one channel in the image). It will output a ‘TrainingData’ folder in your output path which will contain the 640X640 images and their corresponding bounding box information in folders called ‘images’ and ‘labels’ respectively. This function is implemented along with the making of the validation data set in the section labeled 'Generate Training Data'. The validation data is generated using the `MakeValData` function and takes in the path to the training data as well as the number of validation images you want and outputs this data into a 'val' folder containing the validation images and labels. NOTE: if you want to balance your data set in the next step DO NOT generate the validation data set here. To run these functions fill out the variables in the first cells of the 'Generate Training Data' section, run it, and then run the second cell.

5. Once you have generated the training data it is a good idea to check that the conversion of images and mask to 640X640 tiles and bounding boxes has worked properly. To do this you can run the 'CheckBB' function. This function takes in the path to an image and the path to its corresponding label file and plots an image showing the bounding boxes around the nuclei. If the bounding boxes do not look right you will need to troubleshoot the steps above to figure out what went wrong. The ChekBB function is implemented in the section labeled 'Check Training Data'. Below is an example output of the `CheckBB` function.



6. Nuclei can vary a lot in their density, if your training data has a very unbalanced number of images displaying very dense or very sparse nuclei the model may become biased to either call too many or not enough nuclei in an image. To avoid this bias we can try and balance the data using the `FindBalanceData` and `BalanceData` functions. The `FindBalaceData` takes as input the directory to the labels of your training data, how many bins you want to split this data in, and how many images to take per bin. This function determines how many nuclei there are in each image, bins these images depending on how many bins you specify, and then samples the number of files you specified from each bin. The output of the `FindBalaceData` is a list of paths to the labels that make a balanced data set. If you are happy with the number of files that will end up in the balanced data set then you can input this list into the `BalanceData` which will copy the labels and their corresponding images into a new folder named ‘ Balanced’. An example running these functions can be found in the section labeled "Balance Training Data".

7. Once the Balanced data set is established you can run the `MakeValData` function inputting the path to the balanced folder and the number of files you want in your validation data set and the function will output a "val" folder with the images and the labels. An example of this can be found in the last cell of the section labeled "Balance Training Data".

8. Now that we have a training and validation data set we need to fill out a file to tell the model where to look for the data. Go to the data folder by following this path: '/NucID/packages/yolov7/data'. In this folder open up the file called 'Nuclei.yaml' and edit the path next to 'train:' and 'val:' so that they point to your newly generated training and validation data. Save the file and close it.

9. You also have the option to change hyperparameters for training the model. I do not do this and always run with the defaults but the file containing the parameters can be found here: '/NucID/packages/yolov7/data/hyp.scratch.custom.yaml'

10. We are now ready to input the command to train the model. First for the model to train you need your current directory to be the yolov7 package folder (should be this directory: '/NucID/packages/yolov7'). The command for training is run in the terminal (or in the notebook using the correct escape character) and then running `python training.py` followed by the following parameters:

**-- device** Device tells the model to run on the GPU (enter the GPU number ex. 0) or the CPU enter ('cpu'). Since training the model would take way too long on a CPU you will almost always enter `-- device 0`
**--batch-size** This is how many images are used for every forward and backward pass before updating weight. There is a lot online about the costs and benefits of small vs large batch size if you are interested, but I usually use 8 or 16 here and it seems to work well.
**--epochs** This is how many times the model will pass through the whole data set. The more epochs you do the better the model will perform until you start overfitting. I have tried epochs of 50 or 100, not sure if these are optimal but seem to work pretty well.
**--img** this is the dimensions of the input image, the training data should have been made at 640X640 so input 640 here.
**--data** This is the path to the yaml file we edited in step 8, you should enter the path starting from yolov7 ex. 'data/Nuclei.yaml'
**--hyp** These are the hyperparameters the model uses (referred to in step 9) the path should be: ‘data/hyp.scratch.custom.yaml’
**--cfg** The config file contains information about the structure of the model enter this path: ‘cfg/training/yolov7_nuc_cfg.yaml’
**--weights** The model can train starting on blank weights in which case enter '' or you can start off from some pre-trained weights. If you would like to start with pre-trained weights, enter the path to the weights file you want to use ex. '/NucID/weights/10X_ms.pt'
**--name** This will be the name of the folder generated during the training and will contain the newly trained weights and information about the training
**--multi-scale** This is an optional parameter that will occasionally change the scale of images during training to attempt to make the model less sensitive to object sizes

An example of this command can be found in the first cell of the section labeled 'Train your model'

11. The outputs of the training can be found in the following directory: '/NucID/packages/yolov7/runs/train/(name specified in command)/'. This file will contain information about the training session with a nice overview in the 'results.png' file. You can find the weights generated by the model in the 'weights' folder in this directory. I suggest you use the file called 'best.pt' for running the model but it also contains weights generated along the training as well as the weights generated on the last epoch.
12. If the training session gets interrupted for any reason you can resume training by rerunning the training command described in step 10 but deleting the --weights parameter and adding the --resume parameter followed by the path to the last weights generated by the model which are in this directory: '/NucID/packages/yolov7/runs/train/(name specified in command)/weights/last.pt'
13. Once your model is trained you can use the weights generated from it in the "RunNucID.ipynb" to see how it performs. To do so enter the path to the newly generated weights in the 'weights' variable in that file.
