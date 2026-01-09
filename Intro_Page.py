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

if st.button("Shelf Image"):
    st.switch_page("pages/Shelf_Product_upload_v1.py")

if st.button("Planogram Image"):
    st.switch_page("pages/Planogram_Product_upload_v1.py")

if st.button("Final Similarity Check"):
    st.switch_page("pages\Final_Planogram_Comparison_v1.py")