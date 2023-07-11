
__all__ = ['is_cat', 'learn', 'classify_image', 'categories', 'image', 'label', 'examples', 'intf']

# Cell
from fastai.vision.all import *
import gradio as gr

def is_cat(x):
    return x[0].isupper()

# Cell
learn = load_learner("cat_classifier_model.pkl")

# Cell
categories = ("Dog", "Cat")

def classify_img(img):
    pred, index, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Cell
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['KOA_Nassau_2697x1517.jpg', 'beautiful-smooth-haired-red-cat-lies-on-the-sofa-royalty-free-image-1678488026.jpg']

intf = gr.Interface(fn=classify_img, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)
