import os
import logging
import glob
import logging.config
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml
import numpy as np
from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold
from torch.optim.lr_scheduler import StepLR

with open('config.yaml') as file:
  config = yaml.safe_load(file)
logging.config.fileConfig(fname=config["logging_config"], disable_existing_loggers=True)
logger = logging.getLogger(__name__)

from utils import split_data
from dataloader import DocData

os.environ["TOKENIZERS_PARALLELISM"]="false"

logger.info(config)

epochs = config["epochs"]
output_ckpt_path = config["output_ckpt_path"]
output_production_path = config["output_production_path"]
output_logs_path = config["output_logs_path"]
resume = config["resume"]
batch_size = config["batch_size"]
num_labels = config["num_labels"]
full_data_path = config["dataset_path"]
trainset_path = config["train_data_path"]
testset_path = config["test_data_path"]
crossval_folds = config["folds"]

# Split dataset into train/test and place in trainset_path, testset_path
split_data(full_data_path, trainset_path, testset_path)

logger.info(f"Performing {crossval_folds} fold cross-validation")
kf_corssval = KFold(n_splits=crossval_folds)
train_imgs = glob.glob(os.path.join(trainset_path, "**/*.png"))
train_val_splits = list(kf_corssval.split(train_imgs))

os.makedirs(output_ckpt_path, exist_ok=True)
os.makedirs(output_production_path, exist_ok=True)

writer = SummaryWriter(output_logs_path)

if not config["device"]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config["device"]

trainset = DocData(train_imgs)

model = AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=num_labels)
model.to(device)
optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
lr_scheduler = StepLR(optimizer, step_size=config["scheduler_step"], gamma=config["scheduler_decay"])

def checkpoint(model, epoch, step, optimizer, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer,
            "loss": loss,
        },
        path,
    )

def load_checkpoint(save_path):
    # load model weights from model_path
    saved_dict = torch.load(save_path)
    model = saved_dict["model"]
    step = saved_dict["step"]
    optimizer = saved_dict["optimizer"]
    epoch = saved_dict["epoch"]
    loss = saved_dict["loss"]

    logger.info("model loaded from " + save_path)

    return model, optimizer, epoch, step, loss

def train(trainloader, step):
    with tqdm(trainloader, unit="image") as trainloader:
        trainloader.set_description(f"Train Epoch {epoch}")
        for batch in trainloader:
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar("train_loss", loss.item(), step)
            step += 1
            trainloader.set_postfix(
            loss=loss.item(),
            )
        return step

def validate(valloader, step):
    with torch.no_grad():
        with tqdm(valloader, unit="image") as valloader:
            valloader.set_description(f"Validating Epoch {epoch}")
            for encoding in valloader:
                encoding.to(device)
                outputs = model(**encoding)
                loss = outputs.loss
                logits = outputs.logits
                labels = encoding["labels"].detach().cpu().tolist()
                writer.add_scalar("val_loss", loss.item(), step)

                logits = torch.nn.functional.softmax(logits, dim=-1)
                logits = torch.argmax(logits, dim=-1)
                logits = logits.detach().cpu().tolist()

                y.extend(labels)
                y_hat.extend(logits)
                step += 1
            return y, y_hat, step

train_step = 0
val_step = 0
best_f1 = 0

if resume:
    model_dict, optimizer, start_epoch, train_step, loss = load_checkpoint(os.path.join(output_ckpt_path, "last_model.pth"))
    model.load_state_dict(model_dict)
else:
    start_epoch = 1

for epoch in range(start_epoch, epochs):
    y = []
    y_hat = []

    train_indices, val_indices = train_val_splits[epoch % kf_corssval.get_n_splits()]

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    valloader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    train_step = train(trainloader, train_step)
    y, y_hat, val_step = validate(valloader, val_step)
    lr = lr_scheduler.get_last_lr()[-1]
    lr_scheduler.step()
    writer.add_scalar("Learning Rate", lr, epoch)

    assert len(y) == len(y_hat)
    val_report = classification_report(y, y_hat, output_dict=True, zero_division=0)

    curr_f1 = val_report["macro avg"]["f1-score"]
    writer.add_scalar("F1 score", curr_f1, epoch)
    logger.info(f"f1 score = {curr_f1}")
    if curr_f1 >= best_f1:
        pth = f"{output_ckpt_path}/best_model.pth"

        logger.info(f"Saving Best checkpoint {best_f1} ---> {curr_f1}")
        best_f1 = curr_f1
        checkpoint(model, epoch, train_step, optimizer, best_f1, pth)
        torch.save(model.state_dict(), f"{output_production_path}/trained_model.pth")

    pth = f"{output_ckpt_path}/last_model.pth"
    checkpoint(model, epoch, train_step, optimizer, best_f1, pth)
