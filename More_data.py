import SimpleITK as sitk  
from numpy import ctypeslib
import pandas as pd
import numpy as np
import random
import time
import os
import fnmatch
from torch.utils.data import Dataset

class data_processing(Dataset):
    def __init__(self, csv_file_annotation,fake_file_name,csv_file_candidates, root_dir):
        '''
        csv_file = './LUNA16/CSVFILES/annotations.csv'
        csv_file_candidates = './LUNA16/CSVFILES/candidates.csv'
        root_dir = './LUNA16/data'
        '''
        
        self.annotation_csv = pd.read_csv(csv_file_annotation) #, index_col=0 csv_file = './LUNA16/CSVFILES/annotations.csv'
        self.annotation_csv_file_name = csv_file_annotation
        
        self.fake_file_name = fake_file_name
        self.fake = pd.read_csv(fake_file_name)
        
        self.csv_file_candidates_file_name = csv_file_candidates
        self.csv_file_candidates = pd.read_csv(csv_file_candidates)  #

        self.root_dir = root_dir # root_dir = './LUNA16/data'
        self.data_name = []
        for file in os.listdir(self.root_dir):
            if fnmatch.fnmatch(file, '*.mhd'):
                self.data_name.append(file[:-4])
    def remove_candidates_entry_not_in_data(self):
        # get data name for matching
        self.csv_file_candidates \
        = self.csv_file_candidates[self.csv_file_candidates['seriesuid'].isin(self.data_name)]

        header_for_candidates = ["seriesuid", "coordX", "coordY", "coordZ","class"]
        # Write DataFrame to a comma-separated values (csv) file
        self.csv_file_candidates.to_csv(self.csv_file_candidates_file_name, mode='w', columns = header_for_candidates, index=False)

        self.annotation_csv \
        = self.annotation_csv[self.annotation_csv['seriesuid'].isin(self.data_name)]
        header_for_annotation = ["seriesuid", "coordX", "coordY", "coordZ","diameter_mm","class"]
        self.annotation_csv.to_csv(self.annotation_csv_file_name,mode='w',columns = header_for_annotation, index=False)
    def generate_more_data(self,data_n):#參數 data_n 代表要為每一種
        #在 canfidate 內加入新的 dimmeter
        if 'diameter_mm' not in self.csv_file_candidates.columns:
            self.csv_file_candidates.insert(4, "diameter_mm", [0 for i in range(len(self.csv_file_candidates))], True)
            header_for_annotation_1 = ["seriesuid", "coordX", "coordY", "coordZ","diameter_mm","class"]
            self.csv_file_candidates.to_csv(self.csv_file_candidates_file_name\
                                        ,mode='w',columns = header_for_annotation_1, index=False)
        dict_seriesuid = dict()
        # get nodule info through reading annotation file
        for idx in self.annotation_csv.index:
            key = str(self.annotation_csv['seriesuid'][idx])
            cordx1 = float(self.annotation_csv['coordX'][idx])+float(self.annotation_csv['diameter_mm'][idx])
            cordx2 = float(self.annotation_csv['coordX'][idx])-float(self.annotation_csv['diameter_mm'][idx])
            cordy1 = float(self.annotation_csv['coordY'][idx])+float(self.annotation_csv['diameter_mm'][idx])
            cordy2 = float(self.annotation_csv['coordY'][idx])-float(self.annotation_csv['diameter_mm'][idx])
            cordz1 = float(self.annotation_csv['coordZ'][idx])+float(self.annotation_csv['diameter_mm'][idx])
            cordz2 = float(self.annotation_csv['coordZ'][idx])-float(self.annotation_csv['diameter_mm'][idx])
            value = (cordx1,cordx2,cordy1,cordy2,cordz1,cordz2)
            dict_seriesuid.setdefault(key,[])
            dict_seriesuid[key].append(value)
        total_len = len(self.data_name)
        processing_len = 0 
        for scan_name in self.data_name:
            print(str(processing_len)+"/"+str(total_len))
            processing_len +=1
            scan_file_path = self.root_dir +'/'+ scan_name + '.mhd/' + scan_name + '.mhd'
            origin, spacing = self.load_itk_image(scan_file_path)
            file_handle = open(scan_file_path)
            for line in file_handle:
                if line.startswith("DimSize = "):
                    cut_len = len("DimSize = ")
                    dimx,dimy,dimz = tuple(line[cut_len:].split())# dimension of ct scan
                    dimx = int(dimx) # 512 mm
                    dimy = int(dimy) # 512 mm
                    dimz = int(dimz) # 150~200 mm
            # create random number that fall in above range
            # record previous random value to make it more random
            prev_randx,prev_randy,prev_randz=0,0,0
            count = 0
            give_up = 0
            print(scan_name)
            while count < data_n:
                if give_up > 1000000:
                    break
                random.seed(time.time()*1000)
                # 確保產生出來的 x,y,z 座標都落在 ct_scan 內，並針對 x,y,z 三個方向保留 50pixel 的可切割空間
                # 單位 : m.m
                x_virtual = round(random.uniform(origin[2]+51*spacing[2], origin[2]+dimx-51*spacing[2]),2)
                y_virtual = round(random.uniform(origin[1]+51*spacing[1], origin[1]+dimy-51*spacing[1]),2)
                z_virtual = round(random.uniform(origin[0]+51*spacing[0], origin[0]+dimz-51*spacing[0]),2)
                if scan_name not in dict_seriesuid:
                    break
                for cord in dict_seriesuid[scan_name]:
                    #檢查是否落於 nodule 所形成的體積內
                    if(self.if_two_cuboid_intercept(cord[0],\
                                                    cord[1],\
                                                    cord[2],\
                                                    cord[3],\
                                                    cord[4],\
                                                    cord[5],\
                                                    x_virtual,\
                                                    y_virtual,\
                                                    z_virtual,\
                                                    spacing[0],\
                                                    spacing[1],\
                                                    spacing[2])):
                        #print("touch")
                        give_up+=1
                        break
                    if ((prev_randx==x_virtual or prev_randy==y_virtual or prev_randz==z_virtual) and give_up < 3000):
                        give_up+=1
                        break;
                    prev_randx,prev_randy,prev_randz=x_virtual,y_virtual,z_virtual
                    count +=1
                    
                    data = \
                    [{'seriesuid':scan_name,'coordX': x_virtual,'coordY': y_virtual,'coordZ': z_virtual,"diameter_mm":0,'class':0}]
                    
                    self.fake\
                    =self.fake.append(data,ignore_index=True,sort=False)
                    
            file_handle.close()
        header_for_candidates = ["seriesuid", "coordX", "coordY", "coordZ","diameter_mm","class"]
        #fake_filecsv_name = "fake.csv"
        self.fake = pd.concat([self.fake,self.annotation_csv ])
        self.fake = self.fake.sample(frac=1).reset_index(drop=True)
        # in case there are empty entry in class
        for idx in self.fake.index:
            if(pd.isnull(self.fake['class'][idx])):
                self.fake['class'][idx]=1
        self.fake.to_csv(self.fake_file_name, mode='w', columns = header_for_candidates, index=False)
    def load_itk_image(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin()))) # x,y,z  Origin in world coordinates (mm)
        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))
        return  origin, spacing
    def if_two_cuboid_intercept(self,cord0,cord1,cord2,cord3,cord4,cord5,vx,vy,vz,sp0,sp1,sp2):
        dim_of_vx = 50*sp2/2
        dim_of_vy = 50*sp1/2
        dim_of_vz = 50*sp0/2
        if (cord1 <= vx + dim_of_vx <= cord0):
            return True
        if (cord1 <= vx - dim_of_vx <= cord0):
            return True
        if (cord3 <= vy + dim_of_vy <= cord2):
            return True
        if (cord3 <= vy - dim_of_vy <= cord2):
            return True
        if (cord5 <= vz + dim_of_vz <= cord4):
            return True
        if (cord5 <= vz - dim_of_vz <= cord4):
            return True
        return False
#from More_data import data_processing
csv_file = '/home/horovod/LUNA16/CSVFILES/annotations.csv'
csv_file_candidates = '/home/horovod/LUNA16/CSVFILES/candidates.csv'
csv_fake = '/home/horovod/LUNA16/CSVFILES/fake.csv'
root_dir = '/home/horovod/LUNA16/data'
#csv_file_annotation, csv_file_candidates, root_dir
train_set = data_processing(csv_file_annotation = csv_file,
                            fake_file_name = csv_fake,
                            csv_file_candidates = csv_file_candidates ,
                            root_dir = root_dir
                           )

train_set.generate_more_data(5)
