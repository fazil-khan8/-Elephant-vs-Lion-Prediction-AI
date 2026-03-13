import streamlit as st
import torch
from torchvision import models, transforms, datasets
from PIL import Image

# Page configuration
st.set_page_config(page_title="Animal Detection AI", page_icon="🐾", layout="centered")

st.title("🐾 Animal Detection AI")
st.markdown("### Upload an image and let the AI identify the animal")

st.divider()

st.info("""
This AI model can recognize only:

• 🐘 Elephant  
• 🦁 Lion  

Upload a **clear animal image** for better predictions.
""")

# Load dataset classes
dataset = datasets.ImageFolder("dataset")
classes = dataset.classes

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("animal_model.pth", map_location="cpu"))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    prediction = classes[predicted.item()]

    with col2:
        st.subheader("🔍 Prediction")
        st.success(f"**{prediction.upper()}** detected")

st.divider()
st.caption("AI Animal Detection Project • Built with Streamlit & PyTorch")
