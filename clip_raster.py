#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

try:
    from osgeo import ogr
    from osgeo import gdal
except:
    import ogr
    import gdal
import numpy as np
import pandas as pd
import os

import sys
import glob

class ScanFile(object):
    def __init__(self, directory, prefix=None, postfix=None):
        '''
        初始化输入参数
        :param directory: 文件夹路径
        :param prefix:
        :param postfix:
        '''
        
        self.directory = directory
        self.prefix = prefix
        self.postfix = postfix

    def scan_files(self):
        '''
        扫描文件名，获取其路径
        :return: 返回list，里面的各元素是文件名路径
        '''

        files_list = []
        for dirpath, dirnames, filenames in os.walk(self.directory):
            '''''
            dirpath is a string, the path to the directory.
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..').
            filenames is a list of the names of the non-directory files in dirpath.
            '''
            for special_file in filenames:
                if self.postfix:
                    if special_file.endswith(self.postfix): # 判断一个字符的结尾是否是某字符   Python 内置函数 endswith（）
                        files_list.append(os.path.join(dirpath, special_file))
                elif self.prefix:
                    if special_file.startswith(self.prefix): # 判断一个字符的开始是否是某字符   Python 内置函数 endswith（）
                        files_list.append(os.path.join(dirpath, special_file))
                else:
                    files_list.append(os.path.join(dirpath, special_file))

        return files_list

    def scan_subdir(self):
        '''
        扫描子文件夹中文件
        :return: 返回list
        '''
        subdir_list = []
        for dirpath, dirnames, files in os.walk(self.directory):
            subdir_list.append(dirpath)
        return subdir_list

class Main_clip_function(object):


    def clip_func(self, shp_file: str, input_tif: str, output_tif: str, special_nodata_value=-99, data_type='Int16') -> None:
        '''
        该函数调用系统gdal命令实现了利用shapefile文件对tif文件的裁剪

        :param shp_file: 输入的shapefile文件路径
        :param input_tif: 输入的tif文件路径
        :param output_tif: 输出的tif文件路径
        :param special_nodata_value: 设定的nodata值
        :param data_type: 设定的输出tif的数据类型，
        :return: None
        '''

        os.system('gdalwarp -ot %s -dstnodata %s --config GDALWARP_IGNORE_BAD_CUTLINE YES -cutline %s -crop_to_cutline %s %s ' % (data_type, special_nodata_value, shp_file, input_tif, output_tif))

    def rename_shapefile(self, dir_path: str) -> None:
        '''
        该函数实现对目录下的shapefile文件名修改

        :param dir_path: shapefile所在的文件夹路径
        :return: None
        '''

        # 列出当前目录下所有的文件
        files = os.listdir(dir_path)
        for filename in files:
            portion = os.path.basename(filename)
            # 如果后缀是.tif
            if portion.find(' ') != -1:
                # 重新组合文件名和后缀名
                newname = portion.replace(" ", "")
                os.rename(os.path.join(dir_path,filename), os.path.join(dir_path,newname))

    def write_geotiff(self, fname: str, band_r, band_g, band_b,
                      geo_transform, projection, data_type=gdal.GDT_Byte, nodata=0):
        '''
        该函数实现将RGB波段输出为tif影像

        :param fname: 输出的tif文件路径名称
        :param band_r: numpy.array(),红波段的数据
        :param band_g: numpy.array(),绿波段的数据
        :param band_b: numpy.array(),蓝波段的数据
        :param geo_transform: 投影坐标系
        :param projection: 地理坐标系
        :param data_type: gdal数据类型
        :param nodata: 设定输出tif的nodata值
        :return: None
        '''
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = band_r.shape
        outRaster = driver.Create(fname, cols, rows, 3, data_type)
        outRaster.SetGeoTransform(geo_transform)
        outRaster.SetProjection(projection)

        outband1 = outRaster.GetRasterBand(1)
        outband1.WriteArray(band_r)
        outband1.SetNoDataValue(nodata)

        outband2 = outRaster.GetRasterBand(2)
        outband2.WriteArray(band_g)
        outband2.SetNoDataValue(nodata)

        outband3 = outRaster.GetRasterBand(3)
        outband3.WriteArray(band_b)
        outband3.SetNoDataValue(nodata)

        outRaster.FlushCache()

        outRaster = None

    def set_nodata_rgb(self, file: str, output_fname: str) -> None:
        '''
        该函数设定含RGB波段tif影像的nodata

        :param file: 输入的tif影像文件路径
        :param output_fname: 输出的tif影像文件路径
        :return: None
        '''

        dataset = gdal.Open(file, gdal.GA_ReadOnly)
        geo_transform = dataset.GetGeoTransform()
        proj = dataset.GetProjectionRef()

        band_r = dataset.GetRasterBand(1).ReadAsArray()
        band_g = dataset.GetRasterBand(2).ReadAsArray()
        band_b = dataset.GetRasterBand(3).ReadAsArray()

        band_r1 = np.where(band_r == 0, band_r+1, band_r)
        band_g1 = np.where(band_g == 0, band_g+1, band_g)
        band_b1 = np.where(band_b == 0, band_b+1, band_b)

        band_r, band_g, band_b = None, None, None
        del band_r, band_g, band_b

        self.write_geotiff(output_fname, band_r1, band_g1, band_b1, geo_transform,
                      proj)

        band_r1, band_g1, band_b1 = None, None, None
        del band_r1, band_g1, band_b1

    def set_end_nodata(self, input_dir: str, output_dir: str, special_nodata_value=0) -> None:
        '''
        该函数实现对文件夹中所有tif影像设定nodata值

        :param input_dir: 输入的tif影像所在的文件夹路径名
        :param output_dir: 输出的tif影像所在的目标文件名
        :param special_nodata_value: 指定的nodata值
        :return: None
        '''

        tiff_files = glob.glob(os.path.join(input_dir,'*'))
        for i, raster_filename in enumerate(tiff_files):
            out_filename_temp = raster_filename.split('.')[0].split('\\')[-1] + '_out_setnodata.tif'
            out_filename = os.path.join(output_dir, out_filename_temp)
            print('convert %s:' % (i))
            os.system(
                'gdalwarp -dstnodata %s %s %s ' % (
                special_nodata_value, raster_filename, out_filename))

        print('Finish!')


class clip_flow(object):

    @classmethod
    def main(cls):
        # dir = os.getcwd()
        scan_tif = ScanFile(r'K:\2011haidian\source_tif', postfix=".tif")
        # subdirs = scan.scan_subdir()
        tif_files = scan_tif.scan_files()[0]
        print(tif_files)

        main_func = Main_clip_function()
        main_func.rename_shapefile(r'K:\2011haidian\shape\sub_shape')
        scan_geojson = ScanFile(r'K:\2011haidian\shape\sub_shape', postfix=".shp")
        shp_files = scan_geojson.scan_files()

        output_dir = r'K:\2011haidian\output\clip_output'
        for i, shp_filename in enumerate(shp_files):
            out_filename_temp = shp_filename.split('.')[0].split('\\')[-1] + '_out.tif'
            out_filename = os.path.join(output_dir, out_filename_temp)
            print('%s:' % (i))
            # main_func.clip_func(shp_filename, tif_files, out_filename)

        clip_files = glob.glob(os.path.join(output_dir,'*'))

        output_reset_nodata_dir = r'K:\2011haidian\output\reset_nodata'
        for i, raster_filename in enumerate(clip_files):
            out_filename_temp = raster_filename.split('.')[0].split('\\')[-1] + '_out_setnodata.tif'
            out_filename = os.path.join(output_reset_nodata_dir, out_filename_temp)
            print('convert %s:' % (i))
            main_func.set_nodata_rgb(raster_filename, out_filename)

        print('Finish!')


if __name__ == "__main__":
    input_dir = r'H:\output\test7_0302'
    output_dir = r'H:\output\end_0302'
    a = Main_clip_function()
    a.set_end_nodata(input_dir, output_dir)
