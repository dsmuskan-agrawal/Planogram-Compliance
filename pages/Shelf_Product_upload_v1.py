import streamlit as st # type: ignore
import pandas as pd
import numpy as np
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

st.title('Planogram Compliances')
uploaded_file = st.file_uploader("Upload Shelf image required extension jpg",type = ["jpg","png"])

col1, col2 = st.columns([1,1])

def main_image_creation():
    image = Image.open(uploaded_file)
    uploaded_image = cv2.imread(image)
    return uploaded_image

def print_image():
    uploaded_image = main_image_creation()
    with col1:
        if st.button("Show Image"):
            st.image(uploaded_image, caption="Shelf Image for Select Store")


def is_bounding_box_inside(quadrilateral, boxB):
    # Create a Shapely Polygon object for the quadrilateral
    poly = ShapelyPolygon(quadrilateral)
    
    # Create a Shapely box (bounding box) object
    bbox_poly = box(boxB[0], boxB[1], boxB[2], boxB[3])

    # Check if the bounding box lies within the quadrilateral
    return poly.contains(bbox_poly)

def is_product_box_inside(quadrilateral, product):
    boundingBox = product["boundingBox"]
    boxB = [boundingBox['x'], boundingBox['y'], boundingBox['x']+boundingBox['w'], boundingBox['y']+boundingBox['h']]
    print(product["id"])
    return is_bounding_box_inside(quadrilateral, boxB)



def product_detection(quadrilateral):
    img = uploaded_file.getvalue()
    print(type(img))
    uploaded_image = main_image_creation()
    with col2:
        if st.button("Detect Products and Gaps"):
            filtered_products = {}
            run_name = str(uuid.uuid4())
            model_name = 'ms-pretrained-product-detection'
            run = ProductRecognition(run_name, model_name)
            client.create_run(run, img, 'image/png')
            result = client.wait_for_completion(run_name, model_name)
            cv_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            image1, image2 = st.columns(2)
            result_json = json.dumps(result.result, indent=4) 
            sorted_data_values = json.dumps({k: v for k, v in sorted(result_json.items(), key=lambda item: item[1])})
            model_prediction = json.loads(result_json)

            result.result['products'] = [p for p in result.result['products'] if is_product_box_inside(quadrilateral, p)]
            result.result['gaps'] = [p for p in result.result['gaps'] if is_product_box_inside(quadrilateral, p)]
            cv_img_visualized = visualize_recognition_result(result.result, cv_img)
            cv_img_visualized = cv2.cvtColor(cv_img_visualized, cv2.COLOR_BGR2RGB)
            product_detected_image = Image.fromarray(cv_img_visualized) 
            st.image([uploaded_image,product_detected_image], use_column_width=True)

            
    return uploaded_image

    

if uploaded_file is not None:
    print_image()
    uploaded_image = main_image_creation()
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fill color with some transparency
        stroke_width=8,
        stroke_color="#ff0000",
        background_image=uploaded_image,
        update_streamlit=True,
        height=uploaded_image.height,
        width=uploaded_image.width,
        drawing_mode="point",  # Drawing mode is set to point
        key="canvas",
    )

    # Process the results
    if canvas_result.json_data is not None:
        # Extract the points
        points = canvas_result.json_data["objects"]
        quadrilateral = []
        if len(points) == 4:
            coordinates = [(p["left"], p["top"]) for p in points]
            st.write("Coordinates of the 4 points:")
            st.write(coordinates)
            for p in points:
                print("*********************Points are************************** ",(p["left"], p["top"]))
                quadrilateral.append((p["left"], p["top"]))
        else:
            st.write(f"Please click on 4 points. You have clicked on {len(points)} points.")
        product_detection(quadrilateral)