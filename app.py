import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")


INVERT_COLORS = False  


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Handwritten_digit.keras")
model = load_model()

st.title("Handwritten Digit Recognition")
st.write("Draw a digit, then click Predict.")


canvas = st_canvas(
    fill_color="rgba(0,0,0,0)",
    stroke_color="white",
    background_color="black",
    stroke_width=14,
    width=280, height=280,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)
predict_clicked = col1.button("Predict")
clear_clicked = col2.button("Clear")

if clear_clicked:
    st.session_state.clear = True
    st.rerun()

def preprocess(image_rgba: np.ndarray, invert: bool) -> np.ndarray | None:
    """Return (28,28) float32 in [0,1], or None if empty."""
    if image_rgba is None:
        return None
    arr = image_rgba
    if arr.max() <= 1.0:  
        arr = (arr * 255).astype(np.uint8)


    rgb = arr[..., :3].astype(np.float32)
    gray = np.dot(rgb, [0.299, 0.587, 0.114]) / 255.0

 
    if invert:
        gray = 1.0 - gray

    
    fg = gray > 0.2
    if not np.any(fg):
        return None


    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = gray[y0:y1, x0:x1]

    h, w = crop.shape
    s = max(h, w)
    pad_y0 = (s - h) // 2
    pad_x0 = (s - w) // 2
    sq = np.pad(crop,
                ((pad_y0, s - h - pad_y0), (pad_x0, s - w - pad_x0)),
                mode="constant")

    im = Image.fromarray((sq * 255).astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    arr28 = np.asarray(im).astype(np.float32) / 255.0  
    return arr28

if predict_clicked and canvas.image_data is not None:
    arr28 = preprocess(canvas.image_data, INVERT_COLORS)
    if arr28 is None:
        st.warning("No digit detected. Draw and try again.")
    else:
       
        st.image(Image.fromarray((arr28 * 255).astype(np.uint8)).resize((140,140), Image.NEAREST),
                 caption="Model input (28×28)")

        x = arr28.reshape(1, 28, 28, 1).astype(np.float32)
        probs = model.predict(x, verbose=0)           
        pred = int(probs.argmax(axis=1)[0])
        conf = float(probs.max(axis=1)[0])

        st.markdown(f"**Prediction:** {pred}  |  **Confidence:** {conf:.3f}")
