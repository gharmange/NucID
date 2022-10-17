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

class Nucid:
  def __init__(self,nuc_channel,weights_path,package_path,conf_thresh=.4,tileSize=640,overlap=.1, run_um_per_pix=1.29, train_um_per_pix=1.29):
      self.nuc_channel = nuc_channel
      self.weights_path = weights_path
      self.package_path = package_path
      self.conf_thresh = conf_thresh
      self.tileSize = tileSize
      self.overlap = overlap
      self.run_um_per_pix = run_um_per_pix
      self.train_um_per_pix = train_um_per_pix

      self.scale = self.run_um_per_pix/self.train_um_per_pix

      #load the model
      self.LoadModel()

  def LoadModel(self,stride=32):
      #find gpu if it is there or use cpu?
      set_logging()
      self.device = select_device('')
      self.half = self.device.type != 'cpu'
      #load the model then check that the image size is right give max stride for this model
      #figure out why you can change this with variale
      os.chdir(str(self.package_path + '/yolov7'))
      self.model = attempt_load(self.weights_path, map_location=self.device)  # load FP32 model
      self.stride = int(self.model.stride.max())  # model stride
      self.imgsz = check_img_size(self.tileSize, s=self.stride)  # check img_size

      #switch model to cpu if using cpu
      if self.half:
              self.model.half()  # to FP16

      # Run inference (not sure what this does)
      if self.device.type != 'cpu':
          self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
      self.old_img_w = self.old_img_h = self.imgsz
      self.old_img_b = 1

  def LoadImage(self):
      self.image = tifffile.imread(self.tif_path)
      #re-define channels for 0 index
      self.nuc_channel = self.nuc_channel-1

      if len(self.image.shape) == 3:
        self.image = self.image[self.nuc_channel,...]
      else:
          pass

      #make sure image has the correct bit depth
      if self.image.dtype == 'uint16':
          self.image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
      elif self.image.dtype == 'uint8':
          pass
      else:
          print("input files must be in the uint16 or uint8 bit depth")

      #figure our correct scale
      self.scale = self.run_um_per_pix/self.train_um_per_pix
      if self.scale != 1:
        width = self.image.shape[0]
        height = self.image.shape[1]

        scale_width = int(width * self.scale)
        scale_height = int(height * self.scale)

        self.image = cv2.resize(self.image,(scale_width,scale_height))


  def LoadTile(self,tile):
      self.tile = np.array(tile)

      if self.scale != 1:
        new_dim = int(self.tileSize * self.scale)

        self.tile = cv2.resize(self.tile,(new_dim,new_dim))

      tile0 = np.stack([self.tile,self.tile,self.tile],-1)
      assert tile0 is not None, 'Image Not Found '
      self.tile0_shape = tile0.shape

      # Padded resize
      self.tile = letterbox(tile0, 640, stride=32)[0]

      # Convert
      self.tile = self.tile[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
      self.tile = np.ascontiguousarray(self.tile)

      #change format
      self.tile = torch.from_numpy(self.tile).to(self.device)
      self.tile = self.tile.half() if self.half else self.tile.float()  # uint8 to fp16/32
      self.tile /= 255.0  # 0 - 255 to 0.0 - 1.0
      if self.tile.ndimension() == 3:
        self.tile = self.tile.unsqueeze(0)

  def RunModel(self, tile):
    nucyx=[]

    #load tiles
    self.LoadTile(tile)

    # Inference
    pred = self.model(self.tile, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, self.conf_thresh, .45, classes=0, agnostic=True)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(self.tile0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(self.tile.shape[2:], det[:, :4], self.tile0_shape).round()
            # save results (TO DO: make it save confidence
            for *xyxy, conf, cls in reversed(det):
              xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
              nucyx.append([xywh[1]*self.tile0_shape[0], xywh[0]*self.tile0_shape[1],float(conf)])

    if len(nucyx) <= 0:
        nucyx = np.empty(shape=(0, 3))
    else:
      nucyx = np.array(nucyx)

    nucyx[:,0] = nucyx[:,0]/self.scale
    nucyx[:,1] = nucyx[:,1]/self.scale

    self.nucxy = nucyx[:, [1, 0, 2]]


    return Output(nucyx, isimage=False)


  def RunNucID(self, tif_path):
      start = time.time()
      #load image
      self.tif_path = tif_path
      self.LoadImage()
      #put image in deeptile object
      dt = deeptile.load(self.image)

      # Configure
      tile_size = (self.tileSize, self.tileSize)
      overlap_c = (self.overlap, self.overlap)

      # Get tiles
      tiles = dt.get_tiles(tile_size, overlap_c)
      tiles = tiles.pad()

      #load in function to run model
      lifted_f = lift(self.RunModel,vectorized=False)
      #run model on all tiles
      tiled_coords = lifted_f(tiles)
      #stitch coordinates
      stitched_coords = stitch_coords(tiled_coords)
      #change yx to xy
      stitched_coords = stitched_coords[:, [1, 0,2]]
      #scale coordinates to proper image size
      stitched_coords[:,0] = stitched_coords[:,0] #/self.scale
      stitched_coords[:,1] = stitched_coords[:,1] #/self.scale

      #write csv with coordinates of cells
      self.coord_path = self.tif_path.split('.tif')[0] + '_nuc_xy.csv'
      np.savetxt(self.coord_path, stitched_coords , delimiter=",")

      print("Coordintes of Nuclei can be found here: " + self.coord_path)
      #check time it takes
      end = time.time()
      print(end-start)


  def TestModel(self,location,min_brightness=.15,figSize=20):
      self.LoadImage()

      #get tile of interest
      X = location[1]
      Y = location[0]
      half_tile = int(.5 * self.tileSize)
      tile = self.image[X-half_tile:X+half_tile,Y-half_tile:Y+half_tile]

      #run model
      nucyx=[]

      #run model on tile
      self.RunModel(tile)

      dh, dw =  tile.shape
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

      for coord in self.nucxy:
        x, y, conf = coord
        cv2.drawMarker(col_norm_image, (int(x), int(y)),(0,255,0), markerType=cv2.MARKER_CROSS,markerSize=3, thickness=1, line_type=cv2.LINE_AA)
        cv2.putText(col_norm_image, str(round(conf,3)), (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .3, (255,0,0), 1)

      plt.figure(figsize=(figSize, figSize))
      plt.imshow(col_norm_image)
      plt.show()




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


def checkNucXY(tif_path,nucxy_path, conf_thresh = 0, conf_label= True, markerSize=3, min_brightness=.15):
    #load image and coordinates
    image = tifffile.imread(tif_path)
    XY = np.genfromtxt(nucxy_path, delimiter=',')
    XY = XY[XY[:,2] >= conf_thresh]

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

    for coord in XY:
      x, y, conf = coord

      cv2.drawMarker(col_norm_image, (int(x), int(y)),(0,255,0), markerType=cv2.MARKER_CROSS,markerSize=markerSize, thickness=1, line_type=cv2.LINE_AA)
      if conf_label:
          cv2.putText(col_norm_image, str(round(conf,3)), (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,0,255), 1)

    check_path = nucxy_path.split('.csv')[0] + '_check.jpg'
    cv2.imwrite(check_path,col_norm_image)
    print("Image with marked nuclei can be found here: " + check_path)



def FilterCoords(nucxy_path,conf_thresh):
    XY = np.genfromtxt(nucxy_path, delimiter=',')
    XY = XY[XY[:,2] >= conf_thresh]
    new_outpath= nucxy_path.split(".csv")[0] + "_conf_" + str(conf_thresh) + ".csv"
    np.savetxt(new_outpath, XY, delimiter=",")
