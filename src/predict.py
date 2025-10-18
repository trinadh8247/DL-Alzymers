import torch
from PIL import Image
from torchvision import transforms
import io


def preprocess_image(image, input_size=224):
    if isinstance(image, (str, bytes)):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            image = Image.open(io.BytesIO(image)).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)


def predict_from_model(model, image_tensor, device, classes=None, topk=3):
    model = model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().squeeze(0).numpy()

    topk_idx = probs.argsort()[::-1][:topk]
    results = []
    for idx in topk_idx:
        label = classes[idx] if classes else str(idx)
        results.append({"label": label, "probability": float(probs[idx])})
    return results
