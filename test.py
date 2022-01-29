import argparse
import cv2
from mtcnn import MTCNN
from keras.models import Sequential,Model
from keras.layers import Convolution2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.layers.core import Activation, Flatten
from keras.layers.pooling import MaxPooling2D
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

def parse_args():

    parser = argparse.ArgumentParser(description='Test face recognition')

    parser.add_argument('--image_path', help='image to compare with the real time image',
                        required=True, type=str)
    parser.add_argument('--name', help='your name', required=True, type=str)

    args = parser.parse_args()
    return args

def findCosineDistance(source_representation, test_representation):
  a = np.matmul(np.transpose(source_representation), test_representation)
  b = np.sum(np.multiply(source_representation, source_representation))
  c = np.sum(np.multiply(test_representation, test_representation))
  return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def face_comparaison_model():

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    model.load_weights("./models/vgg_face_weights.h5")
    vgg_face_descriptor = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)

    return vgg_face_descriptor

def preprocess_image(image):
    img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def verifyFace(img1,img2_representation,vgg_face_descriptor):
  img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
  
  cosine_similarity = findCosineDistance(img1_representation, img2_representation)
 
  if(cosine_similarity < 0.45):
    return 1
  else:
    return -1

def extract_face(mtcnn_model,img):
    result = mtcnn_model.detect_faces(img)
    if result != []:
        for person in result:
            bounding_box = person['box']
        extracted_face = img[bounding_box[1]: bounding_box[1]+bounding_box[3],bounding_box[0]: bounding_box[0]+ bounding_box[2]]
        return extracted_face
    else: 
        return img 

def main():
    args = parse_args()
    #MTCNN model for face detection
    detector = MTCNN()
    # VGG model for feature extraction
    comparator = face_comparaison_model()
    # Saved image to compare with
    image_to_compare = cv2.imread(args.image_path)
    image_to_compare = extract_face(detector,image_to_compare)
    img2_representation = comparator.predict(preprocess_image(image_to_compare))[0,:]
    cap = cv2.VideoCapture(0)
    while True: 
        #Capture frame-by-frame
        __, frame = cap.read()
    
        #Use MTCNN to detect faces
        result = detector.detect_faces(frame)
        if result != []:
            for person in result:
                bounding_box = person['box']
                cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
            #Crop the detected face from the original image
            detected_face = frame[bounding_box[1]: bounding_box[1]+bounding_box[3],
            bounding_box[0]: bounding_box[0]+ bounding_box[2]]
            #Face comparaison with the saved image
            res = verifyFace(detected_face,img2_representation,comparator)
            # Draw results: Show name if the identity matched else show Not name
            text = args.name if res == 1 else "Not "+args.name
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            # Draw rectangle on the face
            cv2.rectangle(frame, (bounding_box[0], bounding_box[1] - 15), 
            (bounding_box[0] + w, bounding_box[1]), (0,155,255), -1)
            cv2.putText(frame, text, (bounding_box[0], bounding_box[1]),
             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        #display resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    #When everything's done, release capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()