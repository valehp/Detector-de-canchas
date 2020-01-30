import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os , glob, json
import scipy.misc
import numpy.ma as ma
import cv2
from shapely.geometry import Polygon, Point, MultiPolygon
import imageio

#from helpers.coords_from_grid import coords_from_grid


class preprocessing (object):
    """
    Preprocessing class to obtain train/val/test dataset

    Parameters
    ----------
    raster (gdal obj): Gdal object representing an image of the forest
    mask (gdal obj): Gdal object representing an image of segmentated trees
    

    Returns
    -------
    Saved croped images and dictionary of masks in path.
    """
    def __init__(self, raster, mask=None, region="", method="", rodal=None):
        """ 
        Args:
            raster (gdal obj): Gdal object representing an image of the forest
            mask (gdal obj): Gdal object representing an image of segmentated trees
        """
        self.save_file = ""
        self.region = region
        self.raster_ = raster
        self.mask_ = mask
        self.method = method

        if rodal is not None:
            self.polygon = rodal.shapes()
            self.rodal = rodal
        else:
            self.rodal = rodal

        # Gdal Geotransform object allow us to find location in pixel of a spatial coordinate.
        self.geoMatrix1  = raster.GetGeoTransform()
        self.geoMatrix2 = mask.GetGeoTransform()

        self.extent = self.get_extent()

        # raster_to_matrix returns a Numpy matrix of raster and mask
        self.raster_ = self.raster_to_matrix (self.raster_, 'raster').astype(np.uint16)
        self.mask_  = self.raster_to_matrix(self.mask_, 'mask').astype(np.uint16)

    def set_options(self, name, flips, normal, save_file, translation=0):
        """
        Opciones para:
            
            name: poner nombre a la imagen
            flip: hacer flip (reflexion) de la imagen horixontal y verticalmente
            normal: hacer un recorte a la imagen con las coordenadas sin alterarla
            translation: valor entero del id que indica que solo se va a hacer traslacion (nada mas)
            save_file: carpeta en donde guardar los archivos

        """
        self.name = name
        self.flip = flips
        self.normal = normal
        self.translation = translation
        self.save_file = save_file

    def raster_to_matrix (self, raster, type):
        """
        raster_to_matrix convert raster to numpy matrix

        Parameters
        ----------
        raster (Gdal object): Gdal object representing an image of the forest


        Returns
        -------
        Numpy matrix of raster.
        """
        mtx = []
        count = raster.RasterCount
        if count > 3:
            count = 3

        for i in range(1, count+1):
            band = raster.GetRasterBand(i)
            mtx.append(band.ReadAsArray())

        if raster.RasterCount > 1 :
            mtx = np.stack(mtx, axis= -1)
        else:
            mtx = np.array(mtx[0])

        if type == 'mask':
#            mtx[mtx == mtx.max()] = 65535
            mtx[mtx == mtx.min()] = 65535
        return mtx


    def point_in_poly(self, point):
        shpfilePoints = []
        contain = []
        for shape in self.polygon:
            poly = Polygon(shape.points)
            contain.append(poly.contains(point))
        if len(np.unique(contain)) > 1:
            return True
        else:
            return False

    def crop_rasters (self, coord1, coord2, size, subset):
        """
        crop_rasters method recieve spatial coordinates returns croped images of size "size" in path given by "subset"

        Parameters
        ----------
        coord1 (tuple): Spatial coordinate of upper/right corner to be cropped.
        coord2 (tuple): Spatial coordinate of lower/left corner to be cropped.
        size (int): Size of images to be croped using raster. Size must be dividable by 2 at least 6 times (eg. 256, 320, 384...)
        subset: Type of dataset. Subset will be used as a path for images


        Returns
        -------
        Numpy matrix of mask and raster.
        """
        # Suported subsets are 'train', 'validation' and 'test'
        supported_subset = ['train', 'valid', 'test', 'predict']

        if subset not in supported_subset:
            error_message = 'The given subset name is not supported. '
            suggestion_message = 'Use one within {}.'.format(supported_subset)
            raise NotImplementedError(error_message + suggestion_message)

        self.size = size
        self.x1, self.y1 = coord1
        self.x2, self.y2 = coord2

        # world2Pixel convert spatial coordinate to pixel location.
        up, right   = self.world_to_pixel(self.geoMatrix1, self.x1, self.y1)
        down, left  = self.world_to_pixel(self.geoMatrix1, self.x2, self.y2)
        if right > left:
            self.raster = self.raster_[left:right,down:up,:]
        else:
            self.raster = self.raster_[right:left,down:up,:]


        up, right  = self.world_to_pixel(self.geoMatrix2, self.x1, self.y1)
        down, left = self.world_to_pixel(self.geoMatrix2, self.x2, self.y2)
        if right > left:
            self.mask  = self.mask_[left:right,down:up]
        else:
            self.mask  = self.mask_[right:left,down:up]

        if subset not in ['predict']:
            if self.method == 'yolo':
                self.crop_resize_yolo(size, subset)
            elif self.method == 'mask':
                self.crop_resize_mask(size, subset)

        else:
            self.crop_resize_test(size, subset)


    def world_to_pixel(self, geoMatrix, x, y):
        """
        world_to_pixel returns location in pixels of a spatial location.

        Parameters
        ----------
        geoMatrix (Gdal object): Gdal object let us to obtain location in pixels.
        x (float): location in spatial coordinates of x axis.
        y (float): location in spatial coordinates of y axis.


        Returns
        -------
        location in pixels of the spatial coordinate.
        """
        ulX = geoMatrix[0]
        ulY = geoMatrix[3]
        xDist = geoMatrix[1]
        yDist = geoMatrix[5]
        rtnX = geoMatrix[2]
        rtnY = geoMatrix[4]
        pixel = int((x - ulX) / xDist)
        line = int((ulY - y) / xDist)
        return (pixel, line)


    def pixel_to_coord(self, pix_x, pix_y, width, height, coord1, coord2):
        m_x = (coord1[0]-coord2[0])/float(width)
        m_y = (coord2[1]-coord1[1])/float(height)

        coord_y = m_y * (pix_y - height) + coord2[1]
        coord_x = m_x * (pix_x - width) + coord1[0]
        return round(coord_x, 8), round(coord_y, 8)


    def crop_resize_yolo (self, size, subset):
        """
        crop_resize save images and dictionary of mask in subset path.

        Parameters
        ----------
        size (int): Size of images to be croped using raster. Size must be dividable by 2 at least 6 times (eg. 256, 320, 384...)
        subset (str): Type of dataset. Subset will be used as a path for images


        Returns
        -------
        Saved croped images and dictionary of masks in subset path.
        """
        height = self.raster.shape[0]
        width  = self.raster.shape[1]

        self.mask = cv2.resize(self.mask,(width, height), interpolation = cv2.INTER_NEAREST)

        dv_height = 0
        while dv_height + self.size < height:
            dv_width  = 0
            while dv_width + self.size < width:

                msk = self.mask[dv_height:dv_height+self.size, dv_width:dv_width+self.size]

                if len(np.unique(msk)) > 1:

                    try:
                        #f = open("helpers/data/{}/{}.txt".format(self.region, subset), "r+")
                        #contents = f.read()
                        self.w += 1
                        #f.write('helpers/data/{}/images/image_{}.jpg\n'.format(self.region, self.w))
                        #f.close()

                    except:
                        #f = open("helpers/data/{}/{}.txt".format(self.region, subset),"a+")
                        if subset == 'train':
                            self.w = 1
                        else:
                            self.w += 1
                        #f.write('helpers/data/{}/images/image_{}.jpg\n'.format(self.region, self.w))
                        #f.close()

                    self.bbox_annotations(msk, subset)

                    img = self.raster[dv_height:dv_height+self.size, dv_width:dv_width+self.size, :]
                    self.save_image(img, subset)

                dv_width += int(self.size/4)
            dv_height += int(self.size/4)


    def crop_resize_mask (self, size, subset):
        """
        crop_resize save images and dictionary of mask in subset path.

        Parameters
        ----------
        size (int): Size of images to be croped using raster. Size must be dividable by 2 at least 6 times (eg. 256, 320, 384...)
        subset (str): Type of dataset. Subset will be used as a path for images
        

        Returns
        -------
        Saved croped images and dictionary of masks in subset path.
        """

        #mask = scipy.misc.imresize(mask.astype(np.uint16), (raster.shape[0], raster.shape[1]))
        height = self.raster.shape[0]
        width  = self.raster.shape[1]

        #self.mask = cv2.resize(self.mask,(width, height), interpolation = cv2.INTER_NEAREST)      
        """
        try:
            print("aaa")
            #seg_dict = np.load('helpers/data/{}/{}/seg_dict.npy'.format(self.region, subset)).item()
            #keys = seg_dict.keys()
            #self.w = max(list(map(int, keys))) + 1 
        except: 
            print("ohno")
            seg_dict = {}
        #    self.w = 0
        """
        self.w = 0
        dv_height = 0 
        #print("height: {} - width: {}\ndifx: {} - dify: {} || size: {}".format(height, width, abs(self.x1-self.x2), abs(self.y1-self.y2), self.size))
        while dv_height + self.size < height:
            dv_width  = 0
            while dv_width + self.size < width:
                #print("dvw: {} - dvh: {}".format(dv_width, dv_height))
                img = self.raster[dv_height:dv_height+self.size, dv_width:dv_width+self.size, :]
                self.save_image(img, subset)
                self.w += 1
                dv_width += int(self.size/4)
            dv_height += int(self.size/4)


    def crop_resize_test (self, size, subset):
        """
        crop_resize_test save images and dictionary of mask in subset path.

        Parameters
        ----------
        size (int): Size of images to be croped using raster. Size must be dividable by 2 at least 6 times (eg. 256, 320, 384...)
        subset (str): Type of dataset. Subset will be used as a path for images


        Returns
        -------
        Saved croped images and dictionary of masks in subset path.
        """

        #mask = scipy.misc.imresize(mask.astype(np.uint16), (raster.shape[0], raster.shape[1]))
        height = self.raster.shape[0]
        width  = self.raster.shape[1]

        coord1, coord2 = (self.x1, self.y1), (self.x2, self.y2)
        #size_dict = {'height': height, 'width': width, 'coord1': coord1, 'coord2': coord2}

        self.w = 1
        dv_height = 0
        while dv_height + self.size <= height:
            dv_width  = 0
            while dv_width + self.size <= width:
                if self.rodal is not None:
                    x = (dv_width + dv_width + self.size)/2.
                    y = (dv_height + dv_height + self.size)/2.
                    x, y = self.pixel_to_coord(x, y, width, height, coord1, coord2)
                    point = Point(x,y)
                    if self.point_in_poly(point):
                        img = self.raster[dv_height:dv_height+self.size, dv_width:dv_width+self.size, :]
                        self.save_image(img, subset)
                    else:
                        pass
                else:
                    img = self.raster[dv_height:dv_height+self.size, dv_width:dv_width+self.size, :] 
                    self.save_image(img, subset)

                dv_width += int(self.size/2)
                self.w += 1
            dv_height += int(self.size/2)

    def save_image(self, img_array, subset):
        plt.clf()
        plt.imshow(img_array)
        #if self.translation:
        #    scipy.misc.imsave("train/cancha_{}_{}.jpg".format(self.region, self.name), img_array)
        if subset =='train' or subset == 'valid' or subset == 'test':
            #print("save file train: ", self.save_file)
            if self.normal:
                scipy.misc.imsave('{}/cancha_{}_{}_00.jpg'.format(self.save_file, self.region, self.name), img_array)
            if self.flip:
                ud_img_array = np.flipud(img_array)
                lr_img_array = np.fliplr(img_array)
                scipy.misc.imsave('{}/cancha_{}_{}_02.jpg'.format(self.save_file, self.region, self.name), ud_img_array)
                scipy.misc.imsave('{}/cancha_{}_{}_03.jpg'.format(self.save_file, self.region, self.name), lr_img_array)
        elif subset == 'predict':
            scipy.misc.imsave('Predict/cancha_{}_{}.jpg'.format(self.region, self.name), img_array)


    def bbox_annotations(self, msk, subset):
        """
        bbox_annotations returns a dict of boolean masks of each segmentated tree.

        Parameters
        ----------
        seg_dict (dict): empty dictionary.
        msk: array of segmentated trees.
        _id (int): id of the image to be croped.


        Returns
        -------
        Dict of boolean masks of each segmentated tree.
        """

        #f = open("helpers/data/{}/labels/image_{}.txt".format(self.region, self.w),"a+")

        for mask in np.unique(msk):
            if mask != 65535:
                positions = np.where(msk == mask)
                x_max, x_min = np.max(positions[1]), np.min(positions[1])
                width = (x_max - x_min)/self.size
                x_center = (x_max + x_min)/2.
                x_center = x_center/self.size

                y_max, y_min = np.max(positions[0]), np.min(positions[0])
                height = (y_max - y_min)/self.size
                y_center = (y_max + y_min)/2.
                y_center = y_center/self.size

                #f.write("{} {} {} {} {} ".format(0, x_center, y_center, width, height))

        #sf.close()


    def segmentate_arr(self, seg_dict, msk):
        """
        segmentate_arr returns a dict of boolean masks of each segmentated tree.

        Parameters
        ----------
        seg_dict (dict): empty dictionary.
        msk: array of segmentated trees.
        

        Returns
        -------
        Dict of boolean masks of each segmentated tree.
        """

        seg_trees = np.unique(msk)
        i = 0
        for tree in seg_trees:
            if tree < 65000:
                mask = np.zeros((self.size,self.size))
                mask = ma.masked_where(msk == tree, mask)
                mask = ma.getmask(mask)
                seg_dict[self.w][i] = mask
                i+=1
        if seg_trees.shape[0] == 1:
            mask = np.zeros((self.size,self.size)).astype('bool')
            seg_dict[self.w][i] = mask
        return seg_dict


    def get_extent(self):
        geoTransform = self.geoMatrix1

        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * self.raster_.RasterXSize
        miny = maxy + geoTransform[5] * self.raster_.RasterYSize

        return [minx, miny, maxx, maxy]


    def save_img_size(self, region_name):
        labels_path = 'helpers/data/{}/train.txt'.format(region_name)
        paths = np.loadtxt(labels_path, dtype='str')
        widths = []
        heights = []
        for path in paths:
            label_path = path.replace('images','labels').replace(".jpg", ".txt")
            labels = np.loadtxt(label_path).reshape(-1, 5)
            c_x = labels[:,1]
            c_y = labels[:,2]
            ixs = np.where((c_x > 0.25) & (c_x < 0.75) & (c_y > 0.25) & (c_y < 0.75))
            if len(ixs[0]) > 0:
                widths.extend(labels[ixs][:,3])
                heights.extend(labels[ixs][:,4])
        min_size = np.min([np.min(widths), np.min(heights)])
        img_size = int((-(-1//min_size))*32)
#        np.save('helpers/data/{}/img_size.npy'.format(region_name), img_size)




def main_prep(region_name, status, method, predict_exist, if_rodal=False):

    status.append('Loading data...')
    print("Loading data...")

    """
    path_raster = 'helpers/predios/'
    file_raster = path_raster + region_name + '.tif'
    raster = gdal.Open(file_raster)

    path_mask = 'helpers/etiquetas/'
    file_mask = path_mask + region_name + '_etiquetas.tif'
    mask = gdal.Open(file_mask)
    """

    if if_rodal == True:
        path_rodal = 'helpers/rodal/'
        file_rodal = path_rodal + region_name + '_rodal.shp'
        rodal = shapefile.Reader(file_rodal)
    """
    directory = 'helpers/data/{}'.format(region_name)
    os.makedirs(directory, exist_ok=True)
    os.makedirs(directory+'/images', exist_ok=True)
    os.makedirs(directory+'/labels', exist_ok=True)
    os.makedirs('helpers/data/predict_{}'.format(region_name), exist_ok=True)
    
    f = open("helpers/config/{}.txt".format(region_name),"w")
    f.write('classes=1\ntrain=helpers/data/{}/train.txt\nvalid=helpers/data/{}/valid.txt\ntest=helpers/data/{}/test.txt\nnames=helpers/data/{}/classes.txt\n'
             .format(region_name, region_name, region_name, region_name))
    f.close()

    f = open("helpers/data/{}/classes.txt".format(region_name),"w")
    f.write('Tree\n')
    f.close()
    """
    if if_rodal == True:
        prep = preprocessing(raster, mask , region_name, method, rodal)
    else:
        prep = preprocessing(raster, mask , region_name, method)

    #coords, ids = coords_from_grid(grid_name = region_name + "_grilla.shp")
    coords = []
    ids = []
    for index in canchas.index:
        ids.append(index)
        cancha = canchas[index]
        centro = cancha.centroid
        minx, miny = centro.x-1200 , centro.y-1200
        maxx, maxy = centro.x+1200 , centro.y+1200
        coords.append( ( maxx, miny, minx, maxy ) )

    ix = np.random.randint(len(coords), size=1)
    coords_test = coords[ix][0]
    fid_subsets = {'id_test': ids[ix]}
    coords = np.delete(coords, ix, 0) #; del(ids[ix])


    ix = np.random.randint(len(coords), size=1)
    coords_val = coords[ix][0]
    fid_subsets = {'id_val': ids[ix]}
    coords = np.delete(coords, ix, 0) #; del(ids[ix])

    status.append('Preprocessing training data...')
    for coord in coords:
        coord1 = coord[0], coord[1]
        coord2 = coord[2], coord[3]
        prep.crop_rasters(coord1, coord2, 256, 'train')

    fid_subsets = {'id_val': ids}

    #with open('helpers/data/{}/fid_subsets.json'.format(region_name), 'w') as json_file:
    #    json.dump(fid_subsets, json_file)

    status.append('Preprocessing validation data...')

    coord1 = coords_val[0], coords_val[1]
    coord2 = coords_val[3], coords_val[4]
    prep.crop_rasters (coord1, coord2, 256, 'valid')

    status.append('Preprocessing test data...')

    coord1 = coords_test[0], coords_test[1]
    coord2 = coords_test[3], coords_test[4]
    prep.crop_rasters (coord1, coord2, 256, 'test')

    prep.save_img_size(region_name)

    status.append('Preprocessing data for predictions...')

    if not predict_exist:
        raster_extent = prep.extent
        coord1 = raster_extent[2], raster_extent[3]
        coord2 = raster_extent[0], raster_extent[1]
        prep.crop_rasters (coord1, coord2, 256, 'predict')
