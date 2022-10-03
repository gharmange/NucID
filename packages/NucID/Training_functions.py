import os
import tifffile
import torch
from torchvision.ops import masks_to_boxes
import numpy as np
import glob
import deeptile
import gc
import cv2
from CellQuant.CellQuant import MaskCheck
from skimage import measure
import matplotlib.pyplot as plt
import shutil


#funtion to use images and mask data to generate training data for NucID
def MakeTrainingData(img_path,mask_path,outpath,nuc_channel,num_val,tileSize,overlap,num, upSize=False):
  ## Save image tiles
  #read in image
  image = tifffile.imread(img_path)

  if len(image.shape) == 3:
    image = image[nuc_channel,...]
  else:
      pass

  #make sure image has the correct bit depth (NOTE: if training images look bad change these min max values and need to do it in running functions as well)
  if image.dtype == 'uint16':
      #image = bytescaling(image, 0, 20000, high=255, low=0)
      image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  elif image.dtype == 'uint8':
      pass
  else:
      print("input files must be in the uint16 or uint8 bit depth")

  #put image in deeptile object
  dt = deeptile.load(image)
  # Configure
  tile_size = (tileSize, tileSize)
  overlap = (overlap, overlap)

  # Get tiles
  tiles = dt.get_tiles(tile_size, overlap)
  tiles = tiles.pad()

  #generate path variables
  outpath = str(outpath + "/TrainingData")
  if not os.path.exists(outpath):
    os.mkdir(outpath)
  train_outpath = str(outpath + "/train")
  if not os.path.exists(train_outpath):
    os.mkdir(train_outpath)

  img_out_path = str(train_outpath + "/images/")

  if not os.path.exists(img_out_path):
    os.mkdir(img_out_path)

  img_name = img_path.split('/')[-1].split('.tif')[0]

  for (i, j), tile in np.ndenumerate(tiles):
    tile_path = str(img_out_path + str(num) + img_name + '_' + str(i) + '_' + str(j) + '.tif')
    if upSize:
        tile=cv2.resize(np.array(tile),(640,640))
    tifffile.imwrite(tile_path, tile)

  #delete image information to save memmory
  del(tiles)
  del(dt)
  gc.collect()

  ## Save bounding boxes for tiles
  #load mask
  mask = tifffile.imread(mask_path)
  if len(mask.shape) == 2:
      mask = mask
  elif len(mask.shape) == 3:
      mask = mask[0,...]
  elif len(mask.shape) == 4:
      mask = mask[0,...,0]
  else:
      print("mask has unsupported dimensions")

  # Create DeepTile object
  dt = deeptile.load(mask)
  # Get tiles
  tiles = dt.get_tiles(tile_size, overlap)
  tiles = tiles.pad()

  #delete mask now that it is saved as a tile
  #delete mask tiles to save memory
  del(mask)
  gc.collect()

  #generate path variables
  label_out_path = str(train_outpath + "/labels/")
  if not os.path.exists(label_out_path):
    os.mkdir(label_out_path)

  for (i, j), tile in np.ndenumerate(tiles):
    #convert tile correct format
    tile = np.asarray(tile)

    REGIONS = measure.regionprops(tile)
    X=[]
    Y=[]
    W=[]
    H=[]
    for region in REGIONS:
      y_min,x_min,y_max,x_max = region.bbox

      X.append((x_min+x_max)/(2*tileSize))
      Y.append((y_min+y_max)/(2*tileSize))
      W.append((x_max-x_min)/tileSize)
      H.append((y_max-y_min)/tileSize)

    labels = labels = np.zeros([len(X),1])
    boxes = np.column_stack([labels,X,Y,W,H])
    label_tile_path = str(label_out_path + str(num) + img_name + '_' + str(i) + '_' + str(j) + '.txt')
    np.savetxt(label_tile_path, boxes)

  #delete mask tiles to save memory
  del(tiles)
  del(dt)
  gc.collect()




#function used by MakeTrainingData to take a random subset of training data and use for MakeValidation

def MakeValData(base_path,num_val):
  # make validation path
  val_path = str(base_path + '/val')
  if not os.path.exists(val_path):
      os.mkdir(val_path)

  # make validation image path
  val_images_path = str(val_path + '/images')
  if not os.path.exists(val_images_path):
      os.mkdir(val_images_path)

  # make validation label path
  val_labels_path = str(val_path + '/labels')
  if not os.path.exists(val_labels_path):
      os.mkdir(val_labels_path)




  IMAGES = glob.glob(str(base_path + '/train/images/*.tif'))

  VAL_IMAGES = np.random.choice(IMAGES, num_val, replace=False)

  for val_image in VAL_IMAGES:

    image_name = val_image.split('/')[-1]
    label_name = image_name.split('.tif')[0] + '.txt'
    #set paths for image move
    val_image_old = val_image
    val_image_new = str(val_images_path + '/' + image_name)

    #set paths for label move
    val_label_old = str (base_path + '/train/labels/' + label_name)
    val_label_new = str(val_labels_path + '/' + label_name)

    #move image to val
    os.rename(val_image_old, val_image_new)
    #move labels to val
    os.rename(val_label_old, val_label_new)



#remove bad masks
def FilterMask(mask_path,min_area,max_area, image_path = None, check_mask = True):
    mask = tifffile.imread(mask_path)

    if len(mask.shape) == 2:
        mask = mask
    elif len(mask.shape) == 3:
        mask = mask[0,...]
    elif len(mask.shape) == 4:
        mask = mask[0,...,0]
    else:
        print("mask has unsupported dimensions")

    out = np.copy(mask)
    component_areas = np.bincount(mask.ravel())
    too_small = component_areas <= min_area
    too_big = component_areas >= max_area
    too_small_mask = too_small[mask]
    too_big_mask = too_big[mask]
    out[too_small_mask] = 0
    out[too_big_mask] = 0


    #make it so that mask intensity values are consecutive
    keys = np.sort(np.unique(out))
    values = np.array(range(0,len(keys)))
    # Get argsort indices
    sidx = keys.argsort()
    ks = keys[sidx]
    vs = values[sidx]
    out = vs[np.searchsorted(ks,out)]

    #save filtered mask
    outpath = mask_path.split("_mask.tif")[0] + "_min" + str(min_area) + "_max" + str(max_area) + "_filtered_mask.tif"
    tifffile.imwrite(outpath,out.reshape((1,out.shape[0],out.shape[1])).astype('int32'))

    #check mask
    if check_mask:
        MaskCheck(image_path, outpath, channel=1, min_brightness=.15)

## check that bounding boxes look as expected
def CheckBB(img_path,bb_path,label_size=.5,min_brightness=.15):
  #load image
  image = tifffile.imread(img_path)
  dh, dw = image.shape
  ##contrast image properly
  #normalize tiff image
  norm_image = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  # brighten up image if necessary (code taken from: https://stackoverflow.com/questions/57030125/automatically-adjusting-brightness-of-image-with-opencv)
  cols, rows = norm_image.shape
  brightness = np.sum(norm_image) / (255 * cols * rows)
  ratio = brightness / min_brightness
  if ratio >= 1:
      print("no Scale")
  else:
      # Otherwise, adjust brightness to get the target brightness
      norm_image = cv2.convertScaleAbs(norm_image, alpha = 1 / ratio, beta = 0)
  #convert scaled gray scale image to color
  col_norm_image = cv2.cvtColor(norm_image, cv2.COLOR_GRAY2BGR)


  #load bounding boxes
  BBOXES = np.genfromtxt(bb_path)

  if len(BBOXES.shape) == 2:
    pass
  elif len(BBOXES.shape) == 1:
    BBOXES = BBOXES.reshape(-1,len(BBOXES))
  else:
    print("unexpected number of dimensions in labels (likely ther are non bounding boxes in label file)")
  #determine how many inputs there are for bounding box
  n_inputs = BBOXES.shape[1]

  # determine if the first column is labels (assums there is only one type of object and it has the label 0)
  is_label = np.unique(BBOXES[:,0])[0] == 0

  for bb in BBOXES:

    if n_inputs == 4:
      x, y, w, h = bb
      conf=''
    elif n_inputs == 5 and is_label :
      _,x, y, w, h = bb
      conf=''
    elif n_inputs == 5 and is_label == False :
      x, y, w, h, conf = bb
    elif n_inputs == 6 and is_label :
      _,x, y, w, h, conf = bb
      conf=''
    else:
      print('unexpected number of bounding box inputs')

      # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
      # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
    scale = .1
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    cv2.rectangle(col_norm_image, (l, t), (r, b), (0, 255, 0), 1)
    cv2.putText(col_norm_image, conf, (l, t-3), cv2.FONT_HERSHEY_SIMPLEX, label_size, (255,0,0), 1)

  plt.figure(figsize=(20, 20))
  plt.imshow(col_norm_image)
  plt.show()




#Change 8bit to 16bit images
def bytescaling(data, cmin, cmax, high=255, low=0):
    """
    credit: this funtion modified form old scipy funtion
    Converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255). If the input image already has
    dtype uint8, no scaling is done.
    :param data: 16-bit image data array
    :param cmin: bias scaling of small values (def: data.min())
    :param cmax: bias scaling of large values (def: data.max())
    :param high: scale max value to high. (def: 255)
    :param low: scale min value to low. (def: 0)
    :return: 8-bit image data array
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        high = 255
    if low < 0:
        low = 0
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    cscale = cmax - cmin
    if cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


#Funtion for bining files with different number of bounding boxes
def FindBalanceData(path_list, num_bins, sample_size):

  #get sizes of each file
  file_size = []
  for path in path_list:
    file_size.append(os.path.getsize(path))

  #split data into specified number of bins
  bins = np.histogram(file_size, bins = num_bins)
  #index which files land in which bins
  bin_index = np.digitize(file_size, bins[1])

  PATHS_array = np.array(path_list)

  sub_PATHS = []

  #fore each bin take specified number of files and add to list
  for i in np.unique(bin_index):
    num_in_bin = sum(bin_index==i)
    if num_in_bin < sample_size:
      tmp_sample_size = num_in_bin
    else:
      tmp_sample_size = sample_size
    sub_PATHS.append(np.random.choice(PATHS_array[bin_index==i], tmp_sample_size, replace=False))

  sub_PATHS = np.concatenate(sub_PATHS).ravel().tolist()

  print('The number of training files selected is: ' + str(len(sub_PATHS)))
  return sub_PATHS

 #function that moves output of BinData into a Balanced Data folder
def BalanceData(sub_PATHS):
  #define new directories
  old_label_path = sub_PATHS[0]
  base_path = old_label_path.split('TrainingData/train/labels')
  balance_path = str(base_path[0] + 'Balanced/')
  balance_train_path = str(balance_path + 'train/')
  new_labels_path = str(balance_train_path + 'labels/')

  old_image_path = base_path[0] + 'TrainingData/train/images/'
  new_images_path = str(balance_train_path + "images/")

  #make directories
  if not os.path.exists(balance_path):
        os.mkdir(balance_path)

  if not os.path.exists(balance_train_path):
        os.mkdir(balance_train_path)

  if not os.path.exists(new_labels_path):
        os.mkdir(new_labels_path)

  if not os.path.exists(new_images_path):
        os.mkdir(new_images_path)

  for path in sub_PATHS:
    label_file = path.split('/')[-1]

    old_label = path
    new_label = new_labels_path + label_file

    image_file = label_file.split('.txt')[0] + '.tif'

    old_image = old_image_path + image_file
    new_image = new_images_path + image_file

    #move labels to balanced
    shutil.copy(old_label, new_label)
    #move images to balanced
    shutil.copy(old_image, new_image)
