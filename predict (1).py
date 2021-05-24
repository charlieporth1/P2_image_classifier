import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import argparse

parser = argparse.ArgumentParser(description='to predict the probabilities of flowers')
parser.add_argument('image_path',help="image path")
parser.add_argument('model_path',help="model path")
parser.add_argument('--top_k',type= int ,help="top required probabilities", required = False, default = 3)
parser.add_argument('--category_names',
                    help="file path for a json file to map the plant category names",
                    required = False,
                    default = None)
args = parser.parse_args()
def get_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    # Remapping as Class names have index starting from 1 to 102, whereas the datasets have label indices from 0 to 101
    class_new_names = dict()
    for key in class_names:
        class_new_names[str(int(key)-1)] = class_names[key]
    return class_new_names

def process_img(img):
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img , (224,224))
    img /= 225
    img = img.numpy()
    return img

def predict(img_path, model, top_k):
        img = Image.open(img_path)
        img_array = np.asarray(img)
        processed_img = process_img(img_array)
        processed_img_d = np.expand_dims(processed_img, axis=0)
        prediction = model.predict(processed_img_d)
        probabilities , classes = tf.math.top_k(prediction, top_k)
        return list(probabilities.numpy()[0]) , list(classes.numpy()[0])

def main(image_path , model_path , top_k, category_names):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    prob , classes = predict(image_path , model , top_k)

    print(f'probabilites    = {prob}')
    print(f'classes numbers = {classes}')

    
  

    return prob , classes , classes_names_list





if __name__ == '__main__' :
    if args.category_names != None:
       classes_names_list = get_class_names(args.category_names)
    else:
       classes_names_list = None
        
    main(args.image_path , args.model_path , args.top_k , classes_names_list)
