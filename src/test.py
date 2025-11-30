import os
import glob

import json

import cv2

classes = ['1_Pronacio', '2_Neutralis', '3_Szupinacio']



def main():
    image = cv2.imread("data\\all_data\\C6037J\\internet_freepik_02.jpg")
    print(image.shape)

if __name__ == "__main__":
    main()