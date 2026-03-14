import torch
import timm
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS = {
    "deleted"      : 0.60,
    "adjudication" : 0.60,
    "active"       : 0.70,
    "empty"        : 0.70,
}

infer_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_model(model_path: str):
    # weights_only=True — safe loading, required for PyTorch 2.0+
    checkpoint  = torch.load(model_path, map_location=DEVICE, weights_only=True)
    class_names = checkpoint.get("class_names",
                                 ["active", "deleted", "adjudication", "empty"])
    model = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    print(f"✅ Model loaded on {DEVICE} | Classes: {class_names}")
    return model, class_names, idx_to_class


@torch.no_grad()
def classify_card(pil_img, model, idx_to_class):
    tensor = infer_tf(pil_img).unsqueeze(0).to(DEVICE)
    probs  = torch.softmax(model(tensor), dim=1).squeeze().cpu()
    conf, idx = probs.max(0)
    conf       = conf.item()
    pred_class = idx_to_class[idx.item()]
    raw_class  = pred_class
    threshold  = THRESHOLDS.get(pred_class, 0.85)

    if conf < threshold:
        pred_class = "active"

    prob_dict = {idx_to_class[i]: round(probs[i].item(), 4)
                 for i in range(len(idx_to_class))}

    return pred_class, round(conf, 4), prob_dict, raw_class, (pred_class != raw_class)
