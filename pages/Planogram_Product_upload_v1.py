import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import io
from io import StringIO
from PIL import Image
from PIL import ImageOps
import logging
import uuid
logging.getLogger().setLevel(logging.INFO)
import cv2
from cognitive_service_vision_model_customization_python_samples import ResourceType
from cognitive_service_vision_model_customization_python_samples.clients import ProductRecognitionClient
from cognitive_service_vision_model_customization_python_samples.models import ProductRecognition
from cognitive_service_vision_model_customization_python_samples.tools import visualize_recognition_result
from streamlit_drawable_canvas import st_canvas # type: ignore
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from shapely.geometry import Polygon as ShapelyPolygon, box # type: ignore
import json
from pdf2image import convert_from_bytes # type: ignore
from streamlit_cropper import st_cropper # type: ignore

# Resource and key
resource_type = ResourceType.SINGLE_SERVICE_RESOURCE # or ResourceType.MULTI_SERVICE_RESOURCE

resource_name = None
multi_service_endpoint = None

if resource_type == ResourceType.SINGLE_SERVICE_RESOURCE:
    resource_name = 'mvcomputervision'
    assert resource_name
else:
    multi_service_endpoint = 'https://mvcomputervision.cognitiveservices.azure.com/'
    assert multi_service_endpoint

resource_key = '58ac458beac240deaa68865b48a5dd9e'


client = ProductRecognitionClient(resource_type, resource_name, multi_service_endpoint, resource_key)

st.title('Planogram Compliances V1.0')

# File uploader to upload PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Convert PDF to images
    images = convert_from_bytes(uploaded_file.read())
    page_numbers = list(range(1, len(images) + 1))
    
    # Select page number
    selected_page = st.selectbox("Select page number to crop", page_numbers)
    
    # Display selected page
    image = images[selected_page - 1]
    st.image(image, caption=f'Page {selected_page}', use_column_width=True)
        
    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Crop the image with a resizable box
    st.write("Crop the image using the resizable box:")
    cropped_image = st_cropper(image, realtime_update=True, box_color='#ff0000', aspect_ratio=None)

    if cropped_image:
        crop_img_byte_arr = io.BytesIO()
        cropped_image.save(crop_img_byte_arr, format='PNG')
        img = crop_img_byte_arr.getvalue()
        st.image(cropped_image, caption=f'Cropped Page', use_column_width=True)
        print(type(cropped_image))
        if st.button("Detect Products and Gaps"):
            run_name = str(uuid.uuid4())
            model_name = 'ms-pretrained-product-detection'
            run = ProductRecognition(run_name, model_name)
            client.create_run(run, img, 'image/png')
            result = client.wait_for_completion(run_name, model_name)
            cv_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            image1, image2 = st.columns(2)
            cv_img_visualized = visualize_recognition_result(result.result, cv_img)
            cv_img_visualized = cv2.cvtColor(cv_img_visualized, cv2.COLOR_BGR2RGB)
            product_detected_image = Image.fromarray(cv_img_visualized) 
            st.image([cropped_image,product_detected_image], use_column_width=True)

