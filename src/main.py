import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# 1. Load model (ResNet18, pretrained ImageNet)
model = models.resnet18(pretrained=True)
model.eval()

# 2. Prepare the image
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
x = transform(img).unsqueeze(0)  

# 3. Image
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.strip().split("\n")

# 4. Original image prediction
with torch.no_grad():
    outputs = model(x)
    _, pred = outputs.max(1)
print("[*] Original Image Prediction:", labels[pred.item()])

# 5. FGSM Attacker
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# 6. Calculate Gradient
x.requires_grad = True
outputs = model(x)
init_pred = outputs.max(1)[1]  # shape: (1,)
loss = nn.CrossEntropyLoss()(outputs, init_pred)
model.zero_grad()
loss.backward()
data_grad = x.grad.data

# 7. Into Adversarial image
epsilon = 0.05
x_adv = fgsm_attack(x, epsilon, data_grad)

# 8. Adversarial image prediction
with torch.no_grad():
    outputs_adv = model(x_adv)
    _, pred_adv = outputs_adv.max(1)
print("[*] Adversarial Image Prediction:", labels[pred_adv.item()])

# 9. Visualization
def save_tensor_image(tensor, filename, title=None):
    np_img = tensor.squeeze().detach().permute(1,2,0).numpy()
    plt.imshow(np_img)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# 10. Save original, adversarial images
save_tensor_image(x, "original.png", f"Original: {labels[pred.item()]}")
save_tensor_image(x_adv, "adversarial.png", f"Adversarial: {labels[pred_adv.item()]}")

print("[*] Image Saved: original.png, adversarial.png")
