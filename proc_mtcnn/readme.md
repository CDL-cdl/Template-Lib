## MTCNN Face Landmark Detection
This script uses the MTCNN model to perform face landmark detection on images in a specified directory. It randomly splits the images into a training set and a validation set, and moves the images to their respective folders. The detected face landmarks are saved in text files.
### Dependencies

The script requires the following Python libraries:

- os
- PIL (Pillow)
- facenet_pytorch
- numpy
- random
- shutil

You can install these libraries using pip:

```bash
pip install pillow facenet-pytorch numpy
```

### How to Use  

Specify the directory containing the images in the img_dir variable.
Run the script. The images will be randomly split into a training set and a validation set, and moved to their respective folders. The face landmarks detected by the MTCNN model will be saved in text files in the 'detections' subfolder in each set's folder.
This script is useful for preparing a dataset for training and validating face landmark detection models.

**How to Use(more detailed in chinese)**  

*新建数据集文件夹，与mtcnn_detect.py放在同级路径下*
![Alt text](readme_png\image.png) 
![Alt text](readme_png\image-1.png)

*修改img_dir为数据集文件夹名字*
![Alt text](readme_png\image-3.png)

*修改分为训练集和验证集的名字*
![Alt text](readme_png\image-4.png)

**For example(after)**  
![Alt text](readme_png\image-5.png)
![Alt text](readme_png\image-6.png)
![Alt text](readme_png\image-7.png)