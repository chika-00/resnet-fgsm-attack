# Fooling a ResNet with Adversarial Examples

Ever wondered how to *trick* a image classifier?  
Well, grab some popcorn because we’re about to fool a ResNet-18 into misclassifying a dog with just a tiny bit of pixel magic.

---

## About
This project follows the workflow below:
- Load a **pretrained ResNet-18** from PyTorch
- Run a **normal inference** on an input image
- Craft an **adversarial example** using the Fast Gradient Sign Method (FGSM)
- Compare the predictions **before vs after the attack**
- Save both images as `.png` so you can see the adversarial example image

---

## How to run
1. Clone this repo
2. Install requirements.txt
```bash
pip install torch torchvision matplotlib pillow requests
```
3. Run the script
```bash
python ./src/main.py
```
4. Watch ResNet get confused
5. Check out the images — what ResNet fails to distinguish is still obvious to humans

---

## How does the attack actually work?

 The Fast Gradient Sign Method (FGSM) is quite simple:

The model learns by minimizing a loss function (like cross-entropy).
Normally, gradients from this loss are used to update the weights of the model.

Instead of updating weights, we freeze the model and ask:
“What if we apply the gradient directly to the input image instead?”

By taking the sign of the gradient with respect to the input, we know which direction (per pixel) will increase the loss the most.

- If a pixel should go brighter to confuse the model → +1
- If it should go darker → -1

We then add a very small nudge, scaled by ε (epsilon), to each pixel:

The image looks the same to humans, but to the model, it’s like throwing a banana peel under its feet.
The classifier slips and falls into the wrong label.

