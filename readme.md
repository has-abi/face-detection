# Face Detection
## Face Recognition with MTCNN
```
pip install mtcnn
```
## Feature Extraction With VGG-FACE

Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing) and place it in a directory with /models in the project root folder with the name vgg_face_weights.h5

## Testing

```
python3 test.py --image_path your_image_path --name your_name
```
#### Example
```
python 3 test.py --image_path hassan_abida.png --name Abida
```

