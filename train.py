import os
import logging
import logging.config
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
import random
load_dotenv()

import torch
from transformers import AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR

from dataloader import DocData

logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=True)
logger = logging.getLogger(__name__)
epochs = 100
output_ckpt_path="./models"
output_production_path="./models/weights_only"
output_logs_path = "./logs"
resume = False
batch_size = 2
num_labels = 3

writer = SummaryWriter(output_logs_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
trainset = DocData("train_set")
val_set = DocData("val_set")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=3)
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
# lr_scheduler = StepLR(optimizer, step_size=4, gamma=0.1)
# Split dataset into train/val/test

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

def encoding_to_gpu(encoding):
    for k, v in encoding.items():
        encoding[k] = v.to(device)
    return encoding

def train(trainloader, step):
    with tqdm(trainloader, unit="image") as trainloader:
        trainloader.set_description(f"Epoch {epoch+1}")
        for batch in trainloader:
            optimizer.zero_grad()
            encoding = encoding_to_gpu(batch)
            outputs = model(**encoding)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            writer.add_scalar("train_loss", loss.item(), step)
            step += 1
            trainloader.set_postfix(
            loss=loss.item(),
            )
        return step

def validate(valloader, step):
    with torch.no_grad():
        with tqdm(valloader, unit="image") as valloader:
            valloader.set_description(f"Validating Epoch {epoch+1}")
            for encoding in valloader:
                encoding = encoding_to_gpu(encoding)
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
            
trainloader = DataLoader(trainset, batch_size=batch_size ,shuffle=True)
testloader = DataLoader(val_set, batch_size=batch_size,shuffle=True)
train_step = 0
val_step = 0
best_f1 = 0

if resume:
    model_dict, optimizer, start_epoch, train_step, loss = load_checkpoint(os.path.join(output_ckpt_path, "last_model.pth"))
    model.load_state_dict(model_dict)
else:
    start_epoch = 0

for epoch in range(start_epoch, epochs):
    y = []
    y_hat = []
    train_step = train(trainloader, train_step)
    y, y_hat, val_step = validate(testloader, val_step)
    # lr = lr_scheduler.get_last_lr()[-1]
    # lr_scheduler.step()
    # writer.add_scalar("Learning Rate", lr, epoch+1)
    
    assert len(y) == len(y_hat)
    val_report = classification_report(y, y_hat, output_dict=True, zero_division=0)

    curr_f1 = val_report["macro avg"]["f1-score"]
    writer.add_scalar("F1 score", curr_f1, epoch+1)
    logger.info(f"f1 score = {curr_f1}")
    if curr_f1 > best_f1:
        pth = f"{output_ckpt_path}/best_model.pth"

        logger.info(f"Saving Best checkpoint {best_f1} ---> {curr_f1}")
        best_f1 = curr_f1
        checkpoint(model, epoch + 1, train_step, optimizer, best_f1, pth)
        torch.save(model.state_dict(), f"{output_production_path}/best_model.pth")

    pth = f"{output_ckpt_path}/last_model.pth"
    checkpoint(model, epoch + 1, train_step, optimizer, best_f1, pth)
