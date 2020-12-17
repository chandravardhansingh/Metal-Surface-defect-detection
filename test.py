# Importing libraries
import numpy as np,keras,os,math,cv2,glob,argparse,itertools
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import load_model

#Creating runtime parser for taking values in argument
parser = argparse.ArgumentParser()
parser.add_argument('--path', dest = 'path', default = 'multi', type = str)
args = parser.parse_args()
Path = args.path
if os.path.exists(Path):
    Image_exists = True
else:
    Image_exists = False
    
def MultiPathInference():
    print("Loading model..")
    new_model = load_model('Model.h5')

    get = os.getcwd()
    pos_dir = get + "/img/test/positive"
    neg_dir = get + "/img/test/negative"

    path, dirs, files = next(os.walk(pos_dir))
    file_count = len(files)

    path2, dirs2, files2 = next(os.walk(neg_dir))
    file_count2 = len(files2)

    total_test = file_count + file_count2
    no_epochs = total_test/20
    no_epochs = math.ceil(no_epochs)

    # print(total_test)
    # print(no_epochs)

    test_path = "img/test"

    test_batch = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(test_path,target_size=(224,224),batch_size=20,shuffle=False)

    test_labels=test_batch.classes
    # test_batch.class_indices

    print("Genrating prediction..")
    predections = new_model.predict_generator(test_batch,steps=int(no_epochs),verbose=0)
    # print("Predictions generated...")
    # print(test_labels.shape)
    # print((predections.argmax(axis=1)).shape)
    cm = confusion_matrix(test_labels, predections.argmax(axis=1))

    rgt  = cm[0,0]+cm[1,1]
    total = cm[0,0]+cm[1,1]+cm[1,0]+cm[0,1]
    accuracy = rgt/total
    accuracy = accuracy*100
    return accuracy

def SingleImageInference():
    
    
#     km = glob.glob("*.jpg")[0]
    new_model = load_model('Model.h5')

    immg = cv2.imread(Path)
    immg = cv2.resize(immg,(224,224))
    prdt = new_model.predict(keras.applications.mobilenet.preprocess_input(immg[np.newaxis,:,:,:]))

    mm = prdt.argmax()
    if mm == 0:
        return "Image Doesnt have a Crack"
    if mm == 1:
        return "Image have a Crack"

    
if Path == 'multi':
    print("\t Accuracy = " + str(MultiPathInference()))
    
elif Image_exists == True:
    print(SingleImageInference())
elif Image_exists == False:
    print("Incorrect Image Path...")
