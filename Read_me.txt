MODEL TRAINING GUIDE

Step 1: In terminal, traverse into the Pipeline folder.

            terminal command : cd mcc_mobilenet_pipeline/

Step 2: Activate the tensorflow environment.
    
            terminal command : source activate tensorflow_p36

Step 3: In Pipeline, there are two text files "pos.txt" and "neg.txt".
        Each of them containing corresponding paths for folders containing Positive and Negative Images.

Step 4: Initiate Training

            terminal command : python train.py --pos=pos.txt --neg=neg.txt

It would do following tasks-
1)traverse to the path provided in text files.
2)Collect images from those paths
3)Create Train, validation and test data.
4)Creates an folder name "img/test" in same folder, to store test images from where we can inference in future.  

Step 5: Testing on sample

**For single image testing
    It will tell if the image with given path have crack or not in it.

            terminal command : python test.py --path=/home/ubuntu/mcc_mobilenet_pipeline/n1-Copy1.jpg
              
    Note: Path for image will starts from home directory. for ex."/home/ubuntu/..folder../ImageName.jpg" 
    
**For multi image testing
    It will show the Results in terms of Accuracy.

            terminal command : python test.py

    Note: For inference on multiple new images with same model-
          Replace those images by corrosponding images in "img/test" directory in the same folder.