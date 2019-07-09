from jade.clean_dataset import *
# root_path = "/home/jade/Data/StaticDeepFreeze/2019-03-18_14-11-36"
# test_data = GetVOCTestPath(root_path)
#
# new_path = "/home/jade/Data/StaticDeepFreeze/2019-03-18_14-11-36_0"
# CreateVOCSavePath(new_path)
# for i in range(len(test_data)):
#     name = GetLastDir(test_data[i][:-4])
#     shutil.copy(os.path.join(root_path,DIRECTORY_ANNOTATIONS,name+".xml"),os.path.join(new_path,DIRECTORY_ANNOTATIONS,name+".xml"))
#     shutil.copy(os.path.join(root_path,DIRECTORY_IMAGES,name+".jpg"),os.path.join(new_path,DIRECTORY_IMAGES,name+".jpg"))




CombinedTestVOCImages("/home/jade/Data/StaticDeepFreeze/2019-03-18_14-11-36")