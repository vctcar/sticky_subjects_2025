from datasets import load_dataset

dataset = load_dataset("beans")

from transformers import ViTImageProcessor, ViTForImageClassification
import torch

device = "cuda"

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

model.eval()
model.to(device);

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# First we get the features corresponding to the first training image
encoding = image_processor(images=dataset['train'][0]['image'], return_tensors="pt").to(device)

# Then pass it through the model and get a prediction

######
with torch.no_grad():
  logits = model(**encoding).logits
######

prediction_labels = logits.argmax(-1).item()

print("Predicted class:", model.config.id2label[prediction_labels])

import torch

def transform(example_batch):
    inputs = image_processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs

prepared_ds = dataset.with_transform(transform)

from transformers import AutoModelForImageClassification
import numpy as np
import evaluate

labels = dataset['train'].features['labels'].names

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)},
    ignore_mismatched_sizes=True 
)

metric = evaluate.load("accuracy")

def compute_metrics(sample):
    return metric.compute(
        predictions=np.argmax(sample.predictions, axis=1), 
        references=sample.label_ids)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
  output_dir="./vit-base-beans",  # output directory where the model predictions and checkpoints will be written
  per_device_train_batch_size=16, # batch size
  learning_rate=2e-4,             # learning rate
  num_train_epochs=4,             # number of epochs to train for
  remove_unused_columns=False,    # keep the "image" column
  logging_steps=10,               # how often to print training metrics
  eval_steps=100,                 # how often to measure on the evaluation set
)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"],
    tokenizer=image_processor,
)

train_results = trainer.train()
trainer.save_model("saved_model_files")
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

import datasets
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

dataset = load_dataset("beans")

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