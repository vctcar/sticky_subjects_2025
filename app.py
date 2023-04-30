import datasets
import torch
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

dataset = datasets.load_dataset("beans")

extractor = AutoFeatureExtractor.from_pretrained("saved_model_files")
model = AutoModelForImageClassification.from_pretrained("saved_model_files")

labels = dataset['train'].features['labels'].names


def classify(im):
  features = image_processor(im, return_tensors='pt')
  logits = model(features["pixel_values"])[-1]
  probability = torch.nn.functional.softmax(logits, dim=-1)
  probs = probability[0].detach().numpy()
  confidences = {label: float(probs[i]) for i, label in enumerate(labels)} 
  return confidences


import gradio as gr

interface = gr.Interface(fn=classify, 
                         inputs=gr.inputs.Image(shape=(200, 200)), 
                         outputs="label",
                         title="Bean Classifier",
                         description="Upload a bean to determine the type of disease haunting it",
                         #examples=example
                         )

interface.launch(debug=True)