import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input
)

from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide"
)


# ============================================================
# LOAD EMBEDDINGS AND FILENAMES
# ============================================================

@st.cache_data
def load_data():

    with open("embeddings.pkl", "rb") as f:
        feature_list = np.array(pickle.load(f))

    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)

    return feature_list, filenames


feature_list, filenames = load_data()


# ============================================================
# LOAD RESNET50 MODEL
# ============================================================

@st.cache_resource
def load_model():

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    return model


model = load_model()


# ============================================================
# SAVE UPLOADED FILE
# ============================================================

def save_uploaded_file(uploaded_file):

    try:

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path

    except Exception as e:

        st.error("Unable to save uploaded image.")
        st.exception(e)

        return None


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def feature_extraction(img_path, model):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(
        img_array,
        axis=0
    )

    preprocessed_img = preprocess_input(
        expanded_img_array
    )

    result = model.predict(
        preprocessed_img,
        verbose=0
    ).flatten()

    normalized_result = result / norm(result)

    return normalized_result


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend(features, feature_list):

    neighbors = NearestNeighbors(
        n_neighbors=6,
        algorithm="brute",
        metric="euclidean"
    )

    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors(
        [features]
    )

    return indices, distances


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Settings")

    st.divider()

    st.subheader("👗 Fashion AI")

    st.write(
        "Upload a fashion image and discover "
        "visually similar items."
    )

    st.divider()

    st.subheader("🧠 How it works")

    st.write("**1. 📸 Upload**")
    st.caption("Choose a fashion image.")

    st.write("**2. 🤖 Analyze**")
    st.caption("ResNet50 extracts visual features.")

    st.write("**3. 🔎 Compare**")
    st.caption("The features are compared with your database.")

    st.write("**4. ✨ Recommend**")
    st.caption("The five closest images are displayed.")

    st.divider()

    st.caption(
        "ResNet50 + Nearest Neighbors"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title("👗 Fashion Recommender System")

st.write(
    "Upload a fashion item and find similar products "
    "from your collection."
)

st.divider()


# ============================================================
# FILE UPLOADER
# ============================================================

st.subheader("📸 Find Your Style")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    help="Upload a clear image of a fashion item."
)


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload an image above to get started."
    )

    st.divider()

    st.subheader("✨ What this app does")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            """
            **📸 Upload**

            Choose an image of
            clothing, shoes or accessories.
            """
        )

    with col2:

        st.info(
            """
            **🤖 AI Analysis**

            ResNet50 extracts the
            visual features of your image.
            """
        )

    with col3:

        st.info(
            """
            **👗 Recommendations**

            Find the five most
            visually similar items.
            """
        )

    st.stop()


# ============================================================
# SAVE UPLOADED IMAGE
# ============================================================

file_path = save_uploaded_file(
    uploaded_file
)

if file_path is None:
    st.stop()


# ============================================================
# DISPLAY UPLOADED IMAGE
# ============================================================

st.divider()

st.subheader("🖼️ Your Uploaded Image")

preview_col, info_col = st.columns(
    [1, 2]
)


with preview_col:

    display_image = Image.open(
        uploaded_file
    )

    # IMPORTANT:
    # Do NOT use use_container_width=True here
    # if you want the natural image appearance.

    st.image(
        display_image
    )


with info_col:

    st.write("### Ready to find similar styles?")

    st.write(
        "Our AI will analyze the visual characteristics "
        "of your image and search the fashion database."
    )

    st.info(
        "💡 Tip: Use a clear image where the fashion item "
        "is clearly visible."
    )

    st.write("")

    find_button = st.button(
        "🔍 Find Similar Fashion",
        type="primary"
    )


# ============================================================
# RUN RECOMMENDATION
# ============================================================

if find_button:

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Analyzing your image..."
    ):

        features = feature_extraction(
            file_path,
            model
        )


    # --------------------------------------------------------
    # RECOMMEND
    # --------------------------------------------------------

    with st.spinner(
        "✨ Finding similar fashion..."
    ):

        indices, distances = recommend(
            features,
            feature_list
        )


    st.success(
        "✨ Recommendations found!"
    )

    st.divider()


    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader("📊 Results")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Recommendations",
            "5"
        )

    with metric2:

        st.metric(
            "Best Match Distance",
            f"{distances[0][1]:.3f}"
        )

    with metric3:

        st.metric(
            "Fashion Items",
            len(feature_list)
        )


    st.divider()


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader("✨ Recommended For You")


    # ========================================================
    # IMPORTANT:
    # EXACTLY 5 COLUMNS LIKE YOUR ORIGINAL CODE
    # ========================================================

    col1, col2, col3, col4, col5 = st.columns(5)


    # --------------------------------------------------------
    # IMAGE 1
    # --------------------------------------------------------

    with col1:

        st.image(
            filenames[indices[0][0]]
        )

        st.markdown("### #1")

        st.caption(
            f"Distance: {distances[0][0]:.3f}"
        )


    # --------------------------------------------------------
    # IMAGE 2
    # --------------------------------------------------------

    with col2:

        st.image(
            filenames[indices[0][1]]
        )

        st.markdown("### #2")

        st.caption(
            f"Distance: {distances[0][1]:.3f}"
        )


    # --------------------------------------------------------
    # IMAGE 3
    # --------------------------------------------------------

    with col3:

        st.image(
            filenames[indices[0][2]]
        )

        st.markdown("### #3")

        st.caption(
            f"Distance: {distances[0][2]:.3f}"
        )


    # --------------------------------------------------------
    # IMAGE 4
    # --------------------------------------------------------

    with col4:

        st.image(
            filenames[indices[0][3]]
        )

        st.markdown("### #4")

        st.caption(
            f"Distance: {distances[0][3]:.3f}"
        )


    # --------------------------------------------------------
    # IMAGE 5
    # --------------------------------------------------------

    with col5:

        st.image(
            filenames[indices[0][4]]
        )

        st.markdown("### #5")

        st.caption(
            f"Distance: {distances[0][4]:.3f}"
        )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.divider()

    with st.expander(
        "🧠 How does the recommendation system work?"
    ):

        st.write(
            """
            **ResNet50** is used to extract visual features
            from the uploaded fashion image.

            These features are converted into a numerical
            representation called an embedding.

            The embedding is then compared against the
            embeddings stored in `embeddings.pkl`.

            **Nearest Neighbors** finds the images that have
            the smallest Euclidean distance.

            Therefore:

            **Smaller distance = more visually similar image.**
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "👗 Fashion Recommender System  |  "
    "Powered by ResNet50 + Nearest Neighbors"
)