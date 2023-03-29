import os
import glob
import logging
import warnings
import yaml
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)

from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, LayoutLMv3Config

from dataloader import DocData

os.environ["TOKENIZERS_PARALLELISM"]="false"

with open('config.yaml') as file:
  config = yaml.safe_load(file)

test_imgs = glob.glob(os.path.join(config["test_data_path"], "**/*.png"))

ckpt_path="./weights/trained_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

test_set = DocData(test_imgs)
lmv3_conf = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-base", num_labels=3)
model = AutoModelForSequenceClassification.from_config(lmv3_conf)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
model.to(device)
testloader = DataLoader(test_set, batch_size=1)

y = []
y_hat = []

for encoding in tqdm(testloader):
    encoding.to(device)
    outputs = model(**encoding)
    logits = outputs.logits
    logits = torch.nn.functional.softmax(logits, dim=-1)
    logits = torch.argmax(logits, dim=-1)
    labels = ["email", "resume", "scientific_publication"]
    label = labels[logits.item()]
    y.extend(encoding["labels"].detach().cpu().tolist())
    y_hat.extend(logits.detach().cpu().tolist())

logging.info(classification_report(y, y_hat, output_dict=False, zero_division=0))
