# Name: dispPCAIm.py
# accepts principal components, original file, and shapefile as inputs
# creates raster of principal components in cropped masked region
# specified by shapefile. band order as specified.
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from matplotlib import pyplot as plt

# code used to crop raster using shapefile from
# https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal

def dispPCAIm(PrincComp,infile,shapefile,r,g,b,no_data,outfile):
    # read in shape file and extract polygon information
    shp = gpd.read_file(shapefile)
    geoms = shp.geometry.values
    geoms = [mapping(geoms[0])]

    # open raster file and read in the values in the cropped
    # region, masking regions outside of polygon but inside of
    # rectangular extent
    with rasterio.open(infile) as src:
        out_image, out_transform = mask(src, geoms, crop=True)

    # just the first band. will use this to find unmasked locations
    #data = out_image.data[0]
    data = out_image[0]
    # location of data that are not masked
    row, col = np.where(data != no_data)
    # convert to 1-255 so can be saved in smaller 8-bit unsigned
    # integer format
    Y = PrincComp
    Ymn = np.amin(Y)
    Ymx = np.amax(Y)
    dY = Ymx - Ymn

    Y = (Y - Ymn)/dY*254+1

    print("Principal components have been rescaled to 1:255")
    print("Minimum value before rescaling:",Ymn)
    print("Maximum value before rescaling:",Ymx)

    # numpy array that will eventually be displayed
    P_image = np.copy(out_image)
    P_image[:, row, col] = Y.T

    # open a new raster file for writing
    new_dataset = rasterio.open(
        outfile,
        'w',
        driver = 'GTiff',
        height = P_image.shape[1],
        width = P_image.shape[2],
        count = P_image.shape[0],
        crs = src.crs,
        dtype = src.dtypes[0],
        transform = out_transform,
        nodata = no_data,
    )

    # plot the image using pyplot
    P_py = np.zeros((P_image.shape[1], P_image.shape[2], 3),dtype=np.int)
    rgb = [r,g,b]
    for i in range(0,3):
        P_py[:,:,i] = P_image[rgb[i], :, :]

    plt.imshow(P_py)
    plt.show()

    # write the new file
    new_dataset.write(P_image)
