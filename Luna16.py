#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import SimpleITK as sitk  # 医疗图片处理包
import random


class Luna16(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_segment=True, seg_size=[1, 1, 1]):
        """
        Args:
            mat_file (string): Path to the mat file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotation_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self._is_segment = is_segment
        self.seg_size = np.asarray(seg_size)

    def __len__(self):
        return len(self.annotation_csv['seriesuid'])
        
    def __getitem__(self, idx):
        scan_name = self.annotation_csv['seriesuid'][idx]
        scan_path = self.root_dir + '/' + scan_name + '.mhd/' + scan_name + '.mhd'
        
        # load ct_scan and info
        ct_scan, origin, spacing = self.load_itk_image(scan_path)
        
        # transform to pixel
        ct_scan = ct_scan.astype(np.int32)
        ct_scan = (ct_scan - np.min(ct_scan))/(np.max(ct_scan) - np.min(ct_scan)) * 255
        ct_scan = ct_scan.astype(np.uint8)
        
        # segment nodule
        if self._is_segment:
            nodule, label = self._segment_nodule(idx, ct_scan, origin, spacing)
            # transform
            if self.transform:
                nodule = torch.stack([self.transform(scan_slice) for scan_slice in nodule], dim=1)
            else:
                nodule = np.expand_dims(nodule, axis=0)
            return {'data':nodule, 'label':label}
        else:
            label = self._label(idx, ct_scan, origin, spacing)
            # transform
            if self.transform:
                ct_scan = torch.stack([self.transform(scan_slice) for scan_slice in ct_scan], dim=1)
            else:
                ct_scan = np.expand_dims(ct_scan, axis=0)
            return {'data':ct_scan, 'label':label}
        
        
    def load_itk_image(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))

        return ct_scan, origin, spacing
    
    def _label(self, idx, ct_scan, origin, spacing):
        """
        In mm
        """
        nodule_radius = self.annotation_csv['diameter_mm'][idx] / 2
        # load label z,y,x
        label_coord = np.array([self.annotation_csv['coordZ'][idx], 
                                self.annotation_csv['coordY'][idx], 
                                self.annotation_csv['coordX'][idx]], dtype=np.float32)
        
        label_origin = label_coord - nodule_radius
        
        """
        To pixel
        """
        # transform realworld coord to pixel
        label_origin_pixel = ((label_origin - origin) / spacing).astype(np.int32)
        # label range
        label_segment_range_pixel = (self.annotation_csv['diameter_mm'][idx] / spacing).astype(np.int32) + 1
        
        """
        Mark label
        """
        label = np.zeros(ct_scan.shape, dtype = np.int_) # numpy type int_ to fit label tensor.long
        if(self.annotation_csv['class'][idx] == 1):
            label[label_origin_pixel[0]:label_origin_pixel[0] + label_segment_range_pixel[0],
                  label_origin_pixel[1]:label_origin_pixel[1] + label_segment_range_pixel[1],
                  label_origin_pixel[2]:label_origin_pixel[2] + label_segment_range_pixel[2]] = 255
        
        return label
    
    def _segment_nodule(self, idx, ct_scan, origin, spacing):
        """
        In mm
        """
        nodule_radius = self.annotation_csv['diameter_mm'][idx] * 0.5
        # load label z,y,x
        label_origin = np.array([self.annotation_csv['coordZ'][idx],
                                 self.annotation_csv['coordY'][idx], 
                                 self.annotation_csv['coordX'][idx]], dtype=np.float32)
        
        """
        To pixel
        """
        # transform realworld coord to pixel
        label_origin_pixel = ((label_origin - origin) / spacing).astype(np.int32)
        
        """
        Decide segment range
        """
        # label
        label_segment_range_pixel = ((nodule_radius) / spacing).astype(np.int32) + 1
        # nodule
        nodule_segment_range_pixel = (self.seg_size * 0.5).astype(np.int32) + 1
        # shift
        shift_range_pixle = (nodule_segment_range_pixel - label_segment_range_pixel).astype(np.float32)
        shift_range_pixle = (shift_range_pixle * (random.random() * 2 - 1)).astype(np.int32)
            
        """
        Segment nodule
        """
        label_origin_pixel -= shift_range_pixle
        # avoid nodule out of boundary
        lower_bound, upper_bound = self.boundary(label_origin_pixel, nodule_segment_range_pixel, ct_scan.shape)
        nodule = ct_scan[lower_bound[0]:upper_bound[0],
                         lower_bound[1]:upper_bound[1],
                         lower_bound[2]:upper_bound[2]]
        nodule = nodule.astype(np.uint8)
        
        """
        Mark label
        """
        label = np.zeros(nodule.shape, dtype = np.int_) # numpy type int_ to fit label tensor.long
        if(self.annotation_csv['class'][idx] == 1):
            label_origin_pixel = nodule_segment_range_pixel + shift_range_pixle
            # avoid nodule out of boundary
            lower_bound, upper_bound = self.boundary(label_origin_pixel, label_segment_range_pixel, label.shape)
            label[lower_bound[0]:upper_bound[0],
                  lower_bound[1]:upper_bound[1],
                  lower_bound[2]:upper_bound[2]] = 1
        return nodule, label
    
    def boundary(self, origin, seg_range, size):
        lower_bound = origin - seg_range
        upper_bound = origin + seg_range
        
        # lower boundary
        if(lower_bound[0] < 0):
            origin[0] -= lower_bound[0]
        if(lower_bound[1] < 0):
            origin[1] -= lower_bound[1]
        if(lower_bound[2] < 0):
            origin[2] -= lower_bound[2]
        # upper boundary
        if(upper_bound[0] > size[0]):
            origin[0] -= (upper_bound[0]-size[0])
        if(upper_bound[1] > size[1]):
            origin[1] -= (upper_bound[1]-size[1])
        if(upper_bound[2] > size[2]):
            origin[2] -= (upper_bound[2]-size[2])
            
        lower_bound = origin - seg_range
        upper_bound = origin + seg_range
        return lower_bound, upper_bound
