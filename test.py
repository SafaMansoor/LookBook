import pickle
import numpy as np
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors

model = ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3))

model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D(),
])

feature_list = pickle.load(open('embeddings.pkl','rb'))
print(np.array(feature_list).shape)
filenames = pickle.load(open('filenames.pkl','rb'))

img = image.load_img('sample/shirt.jpeg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img, verbose=0).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0]:
    print(filenames[file])






