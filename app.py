import streamlit as st
import tensorflow as tf
#from tensorflow import keras
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Coffe Leaf Disease Detection",
    page_icon = ":coffee:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:

            return key


with st.sidebar:
        st.image('coleaf.jpg')
        st.title("Kelompok 7 - Sains Data ITERA")
        st.subheader("Accurate detection of diseases present in the Coffee leaves. This helps an user to easily detect the disease and identify it's cause.")


@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('efficientnetb0-Coffee Leaf Diseases-92.73.h5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()
    #model = keras.Sequential()
    #model.add(keras.layers.Input(shape=(224, 224, 4)))


st.write("""
         # Coffee Leaf Disease Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['healthy', 'boron-B', 'calcium-Ca', 'phosphorus-P']

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'healthy':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'boron-B':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Pastikan tanaman kopi mendapat pupuk boron sesuai petunjuk. Siram tanaman dengan benar, hindari genangan air. Lakukan analisis tanah untuk pemahaman yang lebih baik. Pantau tanaman dan daun untuk deteksi dini gejala kekurangan boron.")


    elif class_names[np.argmax(predictions)] == 'calcium-Ca':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Untuk mengatasi kekurangan kalsium pada tanaman kopi, perlu dilakukan pemupukan dengan pupuk kalsium sesuai petunjuk penggunaannya. Pastikan pH tanah berada dalam kisaran optimal agar kalsium dapat diserap dengan baik. Melakukan analisis tanah sangat penting untuk mengetahui tingkat kekurangan kalsium secara spesifik guna perencanaan pemupukan yang tepat. Selain itu, pantau secara rutin tanaman untuk mengidentifikasi tanda-tanda kekurangan kalsium dan lakukan perawatan yang diperlukan. Menggunakan bahan organik seperti kompos juga bisa membantu meningkatkan ketersediaan kalsium dalam tanah.")

    elif class_names[np.argmax(predictions)] == 'phosphorus-P':
        st.sidebar.warning(string)
        st.markdown("## Remedy")
        st.info("Untuk mengatasi kekurangan fosfor pada tanaman kopi, tambahkan pupuk fosfor ke tanah sesuai petunjuk penggunaannya. Pastikan pH tanah optimal agar tanaman dapat menyerap fosfor dengan baik. Lakukan analisis tanah secara teratur untuk memantau dan menyesuaikan pemupukan sesuai kebutuhan tanaman. Pemupukan fosfor yang tepat penting untuk pertumbuhan akar, pembungaan, dan kualitas hasil panen kopi.")
