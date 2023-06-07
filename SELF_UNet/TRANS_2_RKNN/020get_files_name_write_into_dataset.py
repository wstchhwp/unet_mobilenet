#获取一个文件夹里的所有文件名，把文件名写到dataset.txt中
#dataset.txt用于后面量化使用

import os 

def file_name(fileDir):
  count = 0
  f = open("dataset.txt", "w")
  #导入路径
  for root, dirs, files in os.walk(fileDir):
    #获得当前路径下的路径, 文件夹, 文件(list)
    for i in files:
        #循环文件列表
        i = os.path.join(root + "/" + i)
        print("dataset: ",i)
        f.write(i)
        f.write("\n")
  f.close()

file_name("./dataSet_txt/shoudongpodai_imgs")

