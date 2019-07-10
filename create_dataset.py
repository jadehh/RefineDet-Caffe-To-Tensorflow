import os
from jade import *
def create_txt(txtfile,dataset_name,istrain=False):
    with open(txtfile, "r") as lf:
        for line in lf.readlines():
            if istrain:
                filename = line.strip("\n").split(" ")[0]
                img_file = os.path.join(dataset_name, DIRECTORY_IMAGES, filename + ".jpg")
                anno = os.path.join(dataset_name, DIRECTORY_ANNOTATIONS, filename + ".xml")
                with open(txtfile.split(".txt")[0][:-8]+"train"+".txt","a") as f:
                    f.write(img_file+" "+anno+"\n")
            else:
                filename = line.strip("\n")
                img_file = os.path.join(dataset_name, DIRECTORY_IMAGES, filename + ".jpg")
                anno = os.path.join(dataset_name, DIRECTORY_ANNOTATIONS, filename + ".xml")
                print (txtfile.split(".txt")[0][:-3]+"test.txt")
                with open(txtfile.split(".txt")[0][:-3]+"test.txt","a") as f:
                    print (img_file+" "+anno+"\n")
                    f.write(img_file+" "+anno+"\n")


if __name__ == '__main__':
    create_txt("/data2/jdh/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt","VOC2012",istrain=True)
    create_txt("/data2/jdh/VOCdevkit/VOC2012/ImageSets/Main/val.txt","VOC2012")
    # os.rename("/data2/jdh/VOCdevkit/VOC2012/ImageSets/Main/train_val_copy.txt","/data2/jdh/VOCdevkit/VOC2012/ImageSets/Main/test.txt")