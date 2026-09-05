# 👗 Fashion Recommender System

An image-based **Fashion Recommender System** built with **Streamlit,
TensorFlow, ResNet50, and Scikit-learn**.

The application allows a user to upload a fashion image and
automatically finds visually similar items from a pre-existing fashion
image dataset.

------------------------------------------------------------------------

## ✨ Features

-   📸 Upload a fashion image
-   🤖 Extract image features using **ResNet50**
-   🔎 Find visually similar images using **Nearest Neighbors**
-   👗 Display the top 5 recommended fashion items
-   📊 Show similarity distance and database statistics
-   ⚙️ Interactive Streamlit interface
-   ⏳ Loading indicators while the model processes images
-   🧠 Explanation of how the recommendation system works

------------------------------------------------------------------------

## 🛠️ Technologies Used

  Technology           Purpose
  -------------------- --------------------------------------
  Python               Main programming language
  Streamlit            Web application interface
  TensorFlow / Keras   Deep learning and feature extraction
  ResNet50             Pre-trained image feature extractor
  NumPy                Numerical operations
  Pillow               Image processing
  Scikit-learn         Nearest Neighbors similarity search
  Pickle               Store and load image embeddings

------------------------------------------------------------------------

## 🧠 How It Works

The recommendation pipeline has four main steps:

### 1. Upload an Image

The user uploads a fashion image through the Streamlit interface.

### 2. Feature Extraction

The uploaded image is resized to **224 × 224 pixels** and passed through
a pre-trained **ResNet50** model.

The classification layer of ResNet50 is removed, and
`GlobalMaxPooling2D` is used to convert the convolutional features into
a feature vector.

### 3. Similarity Search

The extracted feature vector is normalized and compared with the feature
vectors stored in `embeddings.pkl`.

The application uses:

``` python
NearestNeighbors(
    n_neighbors=6,
    algorithm="brute",
    metric="euclidean"
)
```

A smaller Euclidean distance indicates greater visual similarity.

### 4. Display Recommendations

The five closest fashion images are displayed in a five-column layout.

------------------------------------------------------------------------



## 📊 Recommendation Logic

The system uses Euclidean distance to compare image embeddings.

For two feature vectors:

``` text
distance = Euclidean distance between vectors
```

The nearest images are selected as recommendations.

Example:

``` text
Recommendation #1 → Distance: 0.715
Recommendation #2 → Distance: 0.722
Recommendation #3 → Distance: 0.726
Recommendation #4 → Distance: 0.727
Recommendation #5 → Distance: 0.728
```

A lower distance generally means the images are more visually similar
according to the extracted ResNet50 features.

> Note: The distance value is a similarity metric, not a percentage
> probability.

------------------------------------------------------------------------

## 🎨 User Interface

The Streamlit interface contains:

-   Fashion Recommender title
-   Image upload section
-   Uploaded-image preview
-   "Find Similar Fashion" button
-   Recommendation grid
-   Similarity distances
-   Result statistics
-   Expandable explanation of the AI pipeline
-   Sidebar containing instructions and project information

------------------------------------------------------------------------


## 🔮 Future Improvements

Possible improvements include:

-   🏷️ Fashion category classification
-   🎨 Color-based filtering
-   👟 Separate recommendations for shoes, shirts, pants, etc.
-   🔥 Hybrid recommendations using both visual similarity and metadata
-   ❤️ Save favorite recommendations
-   🛒 Add product links
-   🔐 User accounts
-   📱 Improved mobile interface
-   ⚡ Faster similarity search using FAISS
-   ☁️ Deployment to Streamlit Community Cloud
-   🧠 Fine-tuning the CNN specifically for fashion images

------------------------------------------------------------------------

## 📌 Limitations

The current system primarily measures **visual similarity**.

Two items may look similar but differ in:

-   Brand
-   Price
-   Size
-   Material
-   Availability
-   Actual fashion category

The model is based on ResNet50 features trained on ImageNet, so it is
not specifically fine-tuned for fashion recommendation.

------------------------------------------------------------------------





## ⭐ Acknowledgements

-   TensorFlow / Keras for the ResNet50 implementation
-   Scikit-learn for Nearest Neighbors
-   Streamlit for the interactive web application framework
-   Pillow for image processing
