import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import io


@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 7)
    model.load_state_dict(torch.load('./model/currency_classifier.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

class_names = ['1Hundrednote', '2Hundrednote', '2Thousandnote', '5Hundrednote', 'Fiftynote', 'Tennote', 'Twentynote']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("💵 Currency Note Recognition App")


option = st.sidebar.selectbox("Choose Mode", ["Camera Capture", "Image Upload"])


def predict_image(img: Image.Image):
   
    if img.mode != "RGB":
        img = img.convert("RGB")


    input_tensor = transform(img).unsqueeze(0)

    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    predicted_class = class_names[predicted.item()]
    confidence_percent = confidence.item() * 100
    return predicted_class, confidence_percent


if option == "Camera Capture":
    st.subheader("📷 Capture a Currency Note Using Webcam")

    camera_img = st.camera_input("Take a photo")

    if camera_img is not None:
       
        img = Image.open(camera_img)

        st.image(img, caption="Captured Image", use_container_width=True)

        predicted_class, confidence = predict_image(img)

        if confidence >= 20:
            st.success(f"✅ Prediction: **{predicted_class}** ({confidence:.2f}%)")
        else:
            st.warning(f"⚠️ Low Confidence: ({confidence:.2f}%) — Try Again!")

elif option == "Image Upload":
    st.subheader("📁 Upload an Image of a Currency Note")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        predicted_class, confidence = predict_image(image)

        if confidence >= 20:
            st.success(f"✅ Prediction: **{predicted_class}** ({confidence:.2f}%)")
        else:
            st.warning(f"⚠️ Low Confidence: ({confidence:.2f}%) — Try Again!")
