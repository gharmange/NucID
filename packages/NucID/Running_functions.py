import cv2
import numpy as np
import torch
import os
import time
import tifffile
import matplotlib.pyplot as plt
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, load_classifier
import deeptile
from deeptile import Output
from deeptile.extensions.stitch import stitch_coords
from deeptile import lift

## Funtion to load the model on GPU or CPU
def LoadModel(weights,PackagePath,imgsz,stride=32):
    #find gpu if it is there or use cpu?
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'
    #load the model then check that the image size is right give max stride for this model
    #figure out why you can change this with variale
    os.chdir(str(PackagePath + '/yolov7'))
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    #switch model to cpu if using cpu
    if half:
            model.half()  # to FP16

    # Run inference (not sure what this does)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    return model, device, half, old_img_b, old_img_h, old_img_w




#Function for loading in tiles in the right format to be run in the model

def LoadTile(img,device,half,upSize):
    img = np.array(img)

    img0 = np.stack([img,img,img],-1)
    assert img0 is not None, 'Image Not Found '

    if upSize:
        img=cv2.resize(np.array(img),(640,640))

    # Padded resize
    img = letterbox(img0, 640, stride=32)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    #change format
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
      img = img.unsqueeze(0)
    return img, img0.shape

# Function needed for Loading Tiles

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)



# Function for running the model

def RunModel(tile, model, conf, device, half, old_img_b, old_img_h, old_img_w,upSize):
    nucyx=[]

    #load tiles
    img, img0_shape = LoadTile(tile,device,half,upSize)

    # Warmup
    #if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
    #    old_img_b = img.shape[0]
    #    old_img_h = img.shape[2]
    #    old_img_w = img.shape[3]
    #    for i in range(3):
    #        model(img, augment=True)[0]

    # Inference
    pred = model(img, augment=True)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf, .45, classes=0, agnostic=True)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(img0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_shape).round()
            # save results
            for *xyxy, conf, cls in reversed(det):
              xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
              nucyx.append([xywh[1]*img0_shape[0], xywh[0]*img0_shape[1]])

    if len(nucyx) <= 0:
        nucyx = np.empty(shape=(0, 2))
    else:
      nucyx = np.array(nucyx)

    return Output(nucyx, isimage=False)


# Function for running everything in NucID
def RunNucID(tif_path,nuc_channel,weights,PackagePath,conf=.2,tileSize=640,overlap=.1,upSize=False):
  start = time.time()
  #load the model
  model, device, half, old_img_b, old_img_h, old_img_w = LoadModel(weights,PackagePath,tileSize)
  #re-define channels for 0 index
  nuc_channel = nuc_channel-1
  #load image
  image = tifffile.imread(tif_path)

  if len(image.shape) == 3:
    image = image[nuc_channel,...]
  else:
      pass

  #make sure image has the correct bit depth
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

  ##run each tile though the model and output x y coordinates for each tile
  #import model
  lifted_f = lift(RunModel,vectorized=False)
  #run the model on all tiles
  tiled_coords = lifted_f(tiles, model, conf, device, half, old_img_b, old_img_h, old_img_w,upSize)
  #stitch togetehr coordinates from tiles
  stitched_coords = stitch_coords(tiled_coords)

  #change yx to xy
  stitched_coords = stitched_coords[:, [1, 0]]

  #write csv with coordinates of cells
  coord_path = tif_path.split('.tif')[0] + '_nuc_xy.csv'
  np.savetxt(coord_path, stitched_coords , delimiter=",")

  print("Coordintes of Nuclei can be found here: " + coord_path)
  #check time it takes
  end = time.time()
  print(end-start)

  #return stitched_coords
  return tiled_coords


# Check how well NucID indentifies nuclei
def checkNucXY(tif_path,nucxy_path, markerSize=3, min_brightness=.15):
  #load image and coordinates
  image = tifffile.imread(tif_path)
  XY = np.genfromtxt(nucxy_path, delimiter=',')

  ## change image so looks good in JPG
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

  ## mark xy coordinates on image
  #convert to ineger
  XY = XY.astype('int')

  for item in XY:
      cv2.drawMarker(col_norm_image, (item[0], item[1]),(0,0,255), markerType=cv2.MARKER_CROSS,markerSize=markerSize, thickness=1, line_type=cv2.LINE_AA)

  check_path = nucxy_path.split('.csv')[0] + '_check.jpg'
  cv2.imwrite(check_path,col_norm_image)
  print("Image with marked nuclei can be found here: " + check_path)


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



#This funtion runs a specified model on one tile and shows which nuclei are identified and how confidenlty
def TestModel(tif_path,location,nuc_channel,weights,PackagePath,conf=.2,tileSize=640, min_brightness=.15,figSize=20,upSize=False):
  #load the model
  model, device, half, old_img_b, old_img_h, old_img_w = LoadModel(weights,PackagePath,tileSize)
  #re-define channels for 0 index
  nuc_channel = nuc_channel-1
  #load image
  image = tifffile.imread(tif_path)

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


  #get tile of interest
  X = location[1]
  Y = location[0]
  half_tile = int(.5 * tileSize)
  tile = image[X-half_tile:X+half_tile,Y-half_tile:Y+half_tile]

  #run model
  nucyx=[]

  #load tiles
  img, img0_shape = LoadTile(tile,device,half,upSize)

  # Warmup
  if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
      old_img_b = img.shape[0]
      old_img_h = img.shape[2]
      old_img_w = img.shape[3]
      for i in range(3):
          model(img, augment=True)[0]

  # Inference
  pred = model(img, augment=True)[0]
  # Apply NMS
  pred = non_max_suppression(pred, conf, .1, classes=0, agnostic=True)

  # Process detections
  for i, det in enumerate(pred):  # detections per image
      gn = torch.tensor(img0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
      if len(det):
          # Rescale boxes from img_size to im0 size
          det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0_shape).round()
          # save results
          for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            nucyx.append([xywh[1]*img0_shape[0], xywh[0]*img0_shape[1],float(conf)])

  #change yx to xy
  nucyx = np.array(nucyx)
  nucxy = nucyx[:, [1, 0, 2]]

  dh, dw = tile.shape
  ##contrast image properly
  #normalize tiff image
  norm_image = cv2.normalize(tile, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
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


  for bb in nucxy:
    x, y, conf = bb

    cv2.drawMarker(col_norm_image, (int(x), int(y)),(0,255,0), markerType=cv2.MARKER_CROSS,markerSize=3, thickness=1, line_type=cv2.LINE_AA)
    cv2.putText(col_norm_image, str(round(conf,3)), (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,0,0), 1)

  plt.figure(figsize=(figSize, figSize))
  plt.imshow(col_norm_image)
  plt.show()
