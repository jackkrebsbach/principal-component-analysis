# Name: dispCropIm.py
# Uses a shapefile to crop and mask a raster
# result is saved as a new file and displayed
# using pyplot
import rasterio
import sys
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from matplotlib import pyplot as plt

# code used to crop raster using shapefile from
# https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal

def dispCropIm(infile,shapefile,r,g,b,no_data,outfile):
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
    data = out_image.data[0]

    # location of data that are not masked
    row, col = np.where(data != no_data)
    # number of bands
    bands = out_image.shape[0]

    # extract just the unmasked data
    Y = np.zeros((row.size,bands))
    for ba in range(bands):
        Y[:, ba] = np.extract(data != no_data, out_image.data[ba])

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

    # write the new file
    new_dataset.write(P_image)

    # plot the image using pyplot
    P_py = np.zeros((P_image.shape[1],P_image.shape[2],3),dtype=np.int)
    rgb = [r,g,b]
    for i in range(3):
        P_py[:,:,i] = P_image[rgb[i],:,:]
    plt.imshow(P_py)
    plt.show()
    return(Y)

if __name__ == '__main__':
    infile = sys.argv[1]
    shapefile = sys.argv[2]
    r = int(sys.argv[3])
    g = int(sys.argv[4])
    b = int(sys.argv[5])
    no_data = float(sys.argv[6])
    outfile = sys.argv[7]
    dispCropIm(infile, shapefile, r, g, b, no_data, outfile)

# example %run dispCropIm.py ./m_4208623_nw_16_060_20180707/m_4208623_nw_16_060_20180707.tif ./HopePreserveGMB/HopePreserveGMB.shp 0 1 2 0 ./out_test.tif