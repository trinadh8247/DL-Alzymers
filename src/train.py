import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def train(model, train_loader, val_loader, device, epochs=8, lr=5e-4, checkpoint_path="vgg16_best_valacc.pth"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()*imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.*correct/total if total > 0 else 0.0
        train_loss = running_loss/total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct, val_total, val_running_loss = 0, 0, 0.0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()*imgs.size(0)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_acc = 100.*val_correct/val_total if val_total > 0 else 0.0
        val_loss = val_running_loss/val_total if val_total > 0 else 0.0

        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        try:
            labels_onehot = label_binarize(all_labels, classes=range(model.classifier[6][-1].out_features))
            roc_auc = roc_auc_score(labels_onehot, all_probs, average='macro', multi_class='ovr')
        except Exception:
            roc_auc = 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    print("Training complete. Best Val Acc:", best_val_acc)
    return history
