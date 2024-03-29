import os
import logging
import warnings
from PIL import Image
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
import torch
from transformers import AutoProcessor, AutoModelForSequenceClassification, LayoutLMv3Config

img_path = os.getenv("IMG_PATH")
img = Image.open(img_path)
img = img.convert('RGB')

ckpt_path="./weights/trained_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

lmv3_conf = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-base", num_labels=3)
model = AutoModelForSequenceClassification.from_config(lmv3_conf)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)

model.to(device)
encoding = processor(img, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
encoding.to(device)
outputs = model(**encoding)
logits = outputs.logits
logits = torch.nn.functional.softmax(logits, dim=-1)
logits = torch.argmax(logits, dim=-1)
labels = ["email", "resume", "scientific_publication"]
label = labels[logits.item()]
logging.info(f"The document is a {label}")
