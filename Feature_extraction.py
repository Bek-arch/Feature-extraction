import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image

with open('./model.p', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

img2vec = Img2Vec()

image_path = './data/weather_dataset/val/cloudy/cloudy4.jpg'

image = Image.open(image_path)

image_features = img2vec.get_vec(image)

prediction = loaded_model.predict([image_features])

print(prediction)
