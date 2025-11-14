import streamlit as st 
from PIL import Image


st.set_page_config(page_title="Model Architecture", layout="centered")
st.title("Model Architecture")
col1,col2= st.columns(2)
with col1:
    st.image("https://64.media.tumblr.com/81701565ce6c0176060ddeee21a58604/001095284f73cc2f-d7/s540x810/b3869e1149bdf41872d69dbe8985a4c3eddf68cb.gifv",
          caption="This is what my Conv2D feels like during backprop",use_container_width=True,width=50)
st.markdown("""Before diving straight into technical details,let's see an 
            overview of the model architecture""")

img = Image.open("images/model_architecture.png")
st.image(img,caption="Model architecture overview",use_container_width=True)

st.markdown(""" 
            Now let's see the detailed architecture of each layer in the model.""")

img2 = Image.open("images/model_summary.png")
st.image(img2,caption="Detailed model architecture",use_container_width=True)

st.markdown(""" 
            Detailed Model plot""")

img3 = Image.open("images/Model_plot.png")
st.image(img3,caption="Model Plot",use_container_width=True)
