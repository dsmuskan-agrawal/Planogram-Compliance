from pdf2image import convert_from_bytes # type: ignore
from streamlit_cropper import st_cropper # type: ignore
import streamlit as st # type: ignore
import pandas as pd
import numpy as np
from io import StringIO
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
import logging
import uuid
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
import tempfile
import urllib.request
import cv2 
from streamlit_javascript import st_javascript
import torch
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import SamPredictor
from segment_anything import sam_model_registry
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torchvision import models
from torchvision import transforms
import sys
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import gzip
import io
import os
import urllib.parse
import imageio
from matplotlib.widgets import Button
import git
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as tensorimage
from tqdm.auto import tqdm
from skimage.color import rgb2lab, deltaE_ciede2000

st.set_page_config(layout="wide")

transform = transforms.Compose([
    transforms.PILToTensor()
])

SAM_RESOLUTION = 1024
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

class SAMDetector:
    """Run SAM using a various prompting options."""

    def __init__(self, model_filepath=None, download_weights=False, device="cpu", **kwargs):
        """Initialize SAM model with parameters."""
        self._logger = logging.getLogger(__name__)
        self._logger.info("Create the SAM detector")
        if download_weights:
            with tempfile.NamedTemporaryFile(suffix=".pth") as temporary_file:
                urllib.request.urlretrieve(SAM_URL, temporary_file.name)
                self.sam = sam_model_registry["vit_b"](checkpoint=temporary_file.name)
        else:
            self.sam = sam_model_registry["vit_b"](checkpoint=model_filepath)

        self._device = "cpu"
        if device == "cuda" and torch.cuda.is_available():
            self.sam = self.sam.to("cuda")

        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam, **kwargs)
        self.predictor = SamPredictor(self.sam)

    def predict_on_grid(self, image):
        """Run SAM on an image and use a grid as prompt."""
        return self.mask_generator.generate(image)

    def predict_on_bounding_box(self, image, x_min, y_min, x_max, y_max):  # pylint: disable=too-many-arguments
        """Run SAM on an image and use a bounding box as prompt."""
        self.predictor.set_image(image)

        mask, _, _ = self.predictor.predict(point_coords=None, point_labels=None, box=np.array([[x_min, y_min, x_max, y_max]]), multimask_output=False)

        return mask[0].astype(int)
    
    @staticmethod
    def convert_background_to_white(image, mask, x_min, y_min, x_max, y_max):
        """Convert the background of the image to white except for the masked bounding box."""
        # Create a white background
        white_background = np.ones_like(image) * 255
        
        # Replace the background with the white background
        masked_image = np.where(mask[..., None], image, white_background)

        return masked_image


    @staticmethod
    def remove_background(crop, mask):
        """Remove the background based a binary mask."""
        background = mask == 0

        crop_no_background = crop.copy()
        crop_no_background[background] = 0

        return crop_no_background

    @staticmethod
    def reduce_bounding_box_by_coordinates(crop, mask):
        """Return the area of interest as given by the mask."""
        x_start, y_start, width, height = mask["bbox"]

        x_start = int(x_start)
        y_start = int(y_start)
        x_end = x_start + int(width)
        y_end = y_start + int(height)

        return crop[y_start:y_end, x_start:x_end]

    @staticmethod
    def get_sam_image_patch_coordinates(
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        image_height: int,
        image_width: int,
        resolution: int = 1024,
    ):  # pylint: disable=too-many-arguments
        """Compute the coordinates of the SAM image patch that is used as input to the model."""
        # double resolution until it is larger than the bounding box
        while y_max - y_min > resolution or x_max - x_min > resolution:
            resolution *= 2

        # first make sure the resolution fits within the crop
        resolution = min(resolution, image_height, image_width)

        patch_x_center = (x_max + x_min) // 2
        patch_y_center = (y_max + y_min) // 2

        # center the SAM image patch on the bounding box center
        patch_x_min = patch_x_center - resolution // 2
        patch_y_min = patch_y_center - resolution // 2

        patch_x_max = patch_x_center + resolution // 2
        patch_y_max = patch_y_center + resolution // 2

        # check if the patch bbox is completely within the image resolution
        if patch_x_max > image_width:  # move patch to the left
            patch_x_min -= patch_x_max - image_width
            patch_x_max = image_width

        if patch_x_min < 0:  # move patch to the right
            patch_x_max -= patch_x_min
            patch_x_min = 0

        if patch_y_max > image_height:  # move patch up
            patch_y_min -= patch_y_max - image_height
            patch_y_max = image_height

        if patch_y_min < 0:  # move patch down
            patch_y_max -= patch_y_min
            patch_y_min = 0

        return patch_x_min, patch_y_min, patch_x_max, patch_y_max

    @staticmethod
    def get_sam_patch_bounding_box_coordinates(
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        patch_x_min: int,
        patch_y_min: int,
        patch_height: int,
        patch_width: int,
    ):  # pylint: disable=too-many-arguments
        """Compute the coordinates of the bounding box within the image patch.

        x_min: the x-coordinate of the top-left corner of the bounding box
        y_min: the y-coordinate of the top-left corner of the bounding box
        x_max: the x-coordinate of the bottom-right corner of the bounding box
        y_max: the y-coordinate of the bottom-right corner of the bounding box
        patch_x_min: the x-coordinate of the top-left corner of the context patch surrounding the bounding box
        patch_y_min: the y-coordinate of the top-left corner of the context patch surrounding the bounding box
        patch_height: the height of the patch
        patch_width: the width of the patch
        """
        # compute the coordinates of the top-left corner of the bounding box within the image patch
        patch_bbox_x_min = x_min - patch_x_min
        patch_bbox_y_min = y_min - patch_y_min

        # in case patch is smaller than bbox
        patch_bbox_x_min = max(0, patch_bbox_x_min)
        patch_bbox_y_min = max(0, patch_bbox_y_min)

        # compute the coordinates of the bottom-right corner of the bounding box within the image patch
        patch_bbox_x_max = x_max - x_min + patch_bbox_x_min
        patch_bbox_y_max = y_max - y_min + patch_bbox_y_min

        patch_bbox_x_max = min(patch_width, patch_bbox_x_max)
        patch_bbox_y_max = min(patch_height, patch_bbox_y_max)

        return patch_bbox_x_min, patch_bbox_y_min, patch_bbox_x_max, patch_bbox_y_max


class SAMCropper:
    """Generate masks on a set of tiles given by a tiler object."""

    def __init__(
        self, detector, tiler, min_size_crop=10000, return_without_background=True, return_with_background=False
    ):  # pylint: disable=too-many-arguments
        """Initialize the cropper."""
        self.tiler = tiler
        self.detector = detector
        self.min_size_crop = min_size_crop
        self.return_without_background = return_without_background
        self.return_with_background = return_with_background

    def __iter__(self):  # pylint: disable=too-many-locals
        """Iterate through all tiles, generate masks and aggregate results."""
        for tile_start_y, _, tile_start_x, _, crop in tqdm(self.tiler, "Finding objects using SAM."):
            masks = self.detector.predict_on_grid(crop)

            for mask in masks:
                bbox_x_min, bbox_y_min, bbox_width, bbox_height = mask["bbox"]

                # combine tile coordinates with image coordinates to get global coordinates
                x_start = tile_start_x + bbox_x_min
                x_end = tile_start_x + bbox_x_min + bbox_width
                y_start = tile_start_y + bbox_y_min
                y_end = tile_start_y + bbox_y_min + bbox_height

                size_crop = (x_end - x_start) * (y_end - y_start)

                if size_crop >= self.min_size_crop:
                    # remove background and refine crop using bbox
                    if self.return_without_background:
                        modified_crop = self.detector.remove_background(crop, mask["segmentation"])
                        modified_crop = self.detector.reduce_bounding_box_by_coordinates(modified_crop, mask)
                        yield x_start, x_end, y_start, y_end, modified_crop

                    if self.return_with_background:
                        modified_crop = self.detector.reduce_bounding_box_by_coordinates(crop, mask)
                        yield x_start, x_end, y_start, y_end, modified_crop

def load_model_from_github(
    repository="https://github.com/facebookresearch/dinov2.git",
    commit="44abdbe27c0a5d4a826eff72b7c8ab0203d02328",
    model="dinov2_vits14",
):
    """Extend the torch hub functionality by allowing to load a model from a specific commit on GitHub."""
    with tempfile.TemporaryDirectory() as output_dir:
        repo = git.Repo.clone_from(repository, output_dir)
        repo.git.checkout(commit)

        model = torch.hub.load(output_dir, model, source="local")

    return model

# import utilities
NORMALIZE_MEAN_VALUES = [0.485, 0.456, 0.406]
NORMALIZE_STD_VALUES = [0.229, 0.224, 0.225]


class Embedder(nn.Module):
    """Model that outputs an embedding for a batch of images based on EfficientNet/DINO."""

    def __init__(
        self,
        architecture="B6",
        pretrained=True,
        device="cpu",
        normalize_mean=NORMALIZE_MEAN_VALUES,
        normalize_std=NORMALIZE_STD_VALUES,
    ):  # pylint: disable=too-many-arguments, dangerous-default-value
        """Initialize the Embedder."""
        super().__init__()

        self._device = device
        self._input_resolution = None
        self._select_model(architecture, pretrained=pretrained)
        self.model.classifier = nn.Sequential()

        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std

        self._transforms = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self._input_resolution, self._input_resolution)),
                transforms.ToTensor(),
                transforms.Normalize(self._normalize_mean, self._normalize_std),
            ],
        )

    def _select_model(self, architecture, pretrained):
        """Select architecture and set matching resolution.

        Resolutions are taken from https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
        and https://github.com/facebookresearch/dino.
        """
        weights = None
        if pretrained:
            weights = f"EfficientNet_{architecture}_Weights.DEFAULT"

        input_resolution_map = {
            "B0": 224,
            "B1": 240,
            "B2": 260,
            "B3": 300,
            "B4": 380,
            "B5": 456,
            "B6": 528,
            "B7": 600,
        }

        architecture_fn_map = {
            "B0": models.efficientnet_b0,
            "B1": models.efficientnet_b1,
            "B2": models.efficientnet_b2,
            "B3": models.efficientnet_b3,
            "B4": models.efficientnet_b4,
            "B5": models.efficientnet_b5,
            "B6": models.efficientnet_b6,
            "B7": models.efficientnet_b7,
        }

        if architecture in ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"):
            self._input_resolution = input_resolution_map[architecture]
            self.model = architecture_fn_map[architecture](weights=weights)

        elif architecture == "DINO":
            self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
            self._input_resolution = 224

        elif architecture.startswith("dinov2_"):
            # self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
            self.model = load_model_from_github(
                repository="https://github.com/facebookresearch/dinov2.git",
                commit="44abdbe27c0a5d4a826eff72b7c8ab0203d02328",
            model="dinov2_vitb14_lc",
            )
            self._input_resolution = 224

        else:
            raise ValueError(f"Incorrect architecture ({architecture}). Should be in B0-B7 or DINO or one of the DINOv2 models.")

        self.model = self.model.to(self._device)

    def forward(self, x):
        """Run Embedder."""
        return self.model(x)

    def preprocess(self, image_patches):
        """Preprocess the input data. Should be provided in batch-first or list format."""
        input_data = torch.stack([self._transforms(image_patch) for image_patch in image_patches])
        return input_data

    def predict(self, image_patches):
        """Generate embeddings for the input data. Should be provided in batch-first or list format."""
        if len(image_patches) == 0:
            return []

        image_patches_processed = self.preprocess(image_patches)
        image_patches_processed = image_patches_processed.to(self._device)

        # do model inference and post-processing
        with torch.no_grad():
            embeddings = self.model(image_patches_processed)
            embeddings = embeddings.detach().cpu().numpy().tolist()

        return embeddings

    def predict_single(self, image_patch):
        """Generate embedding for a single image patch."""
        return self.predict([image_patch])[0]

    def load_checkpoint(self, filepath):
        """Load the weights from the filepath."""
        weights = torch.load(filepath)

        if "state_dict" in weights:
            self.load_state_dict(weights["state_dict"], strict=True)
        else:
            self.load_state_dict(weights, strict=True)

        self.model = self.model.to(self._device)

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

# Create a radio button for navigation
page = st.radio("Go to", ( "Upload Site Image","Upload Planogram Pdf","Final Similarity Check"))

# Session state to hold the uploaded file
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Session state to hold the uploaded file
if "shelf_boundingbox" not in st.session_state:
    st.session_state.shelf_boundingbox = None
    
# Session state to hold the PDF CROPPED Image
if "cropped_image" not in st.session_state:
    st.session_state.cropped_image = None

# Session state to hold the PDF bounding box
if "pdf_boundingbox" not in st.session_state:
    st.session_state.pdf_boundingbox = None


# Page 1: Upload Image
if page == "Upload Site Image":
    image_uploaded_file = st.file_uploader("Upload Shelf image required extension jpg",type = ["jpg","png"])
    st.session_state.image_uploaded_file = image_uploaded_file

    col1 = st.columns([1])

    def main_image_creation():
        image = Image.open(image_uploaded_file)
        uploaded_image = ImageOps.exif_transpose(image)
        return uploaded_image

    def print_image():
        uploaded_image = main_image_creation()


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
        return is_bounding_box_inside(quadrilateral, boxB)



    def product_detection(quadrilateral):
        img = image_uploaded_file.getvalue()
        # print(type(img))
        uploaded_image = main_image_creation()
        col1, col2 = st.columns([1,1])
        with col2:
            if st.button("Detect Products and Gaps"):
                filtered_products = {}
                run_name = str(uuid.uuid4())
                model_name = 'ms-pretrained-product-detection'
                run = ProductRecognition(run_name, model_name)
                client.create_run(run, img, 'image/png')
                result = client.wait_for_completion(run_name, model_name)
                cv_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                st.session_state.shelf_cv_image = cv_img
                image1, image2 = st.columns(2)

                result.result['products'] = [p for p in result.result['products'] if is_product_box_inside(quadrilateral, p)]
                result.result['gaps'] = [p for p in result.result['gaps'] if is_product_box_inside(quadrilateral, p)]
                st.session_state.shelf_boundingbox = result.result
                cv_img_visualized = visualize_recognition_result(result.result, cv_img)
                cv_img_visualized = cv2.cvtColor(cv_img_visualized, cv2.COLOR_BGR2RGB)
                product_detected_image = Image.fromarray(cv_img_visualized) 
                st.image([uploaded_image,product_detected_image], use_column_width=True)      
        return uploaded_image

    # Calculate canvas dimensions
    if image_uploaded_file is not None:
        uploaded_image = main_image_creation()
        # Get the window width
        inner_width = st_javascript("await fetch('').then(function(response) { return window.innerWidth; })")
        canvas_width = uploaded_image.width
        canvas_height = uploaded_image.height 
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fill color with some transparency
            stroke_width=8,
            stroke_color="#ff0000",
            background_image=uploaded_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
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

# Page 2: Display the Image on another page
elif page == "Upload Planogram Pdf":
    st.title("Planogram PDF")
    # File uploader to upload PDF
    pdf_uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf_uploaded_file is not None:
        # Convert PDF to images
        pdf = convert_from_bytes(pdf_uploaded_file.read())
        page_numbers = list(range(1, len(pdf) + 1))
        
        # Select page number
        selected_page = st.selectbox("Select page number to crop", page_numbers)
        
        # Display selected page
        image = pdf[selected_page - 1]
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
            st.session_state.cropped_image = crop_img_byte_arr
            img = crop_img_byte_arr.getvalue()
            st.image(cropped_image, caption=f'Cropped Page', use_column_width=True)
            # print(type(cropped_image))
            if st.button("Detect Products and Gaps"):
                run_name = str(uuid.uuid4())
                model_name = 'ms-pretrained-product-detection'
                run = ProductRecognition(run_name, model_name)
                client.create_run(run, img, 'image/png')
                pdf_result = client.wait_for_completion(run_name, model_name)
                st.session_state.pdf_boundingbox = pdf_result.result
                pdf_cv_img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
                st.session_state.pdf_cv_image = pdf_cv_img
                image1, image2 = st.columns(2)
                cv_img_visualized = visualize_recognition_result(pdf_result.result, pdf_cv_img)
                cv_img_visualized = cv2.cvtColor(cv_img_visualized, cv2.COLOR_BGR2RGB)
                product_detected_image = Image.fromarray(cv_img_visualized) 
                st.image([cropped_image,product_detected_image], use_column_width=True)

def ColorDistance(rgb1,rgb2):
    '''d = {} distance between two colors(3)'''
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = sum((2+rm,4,3-rm)*(rgb1-rgb2)**2)**0.5
    return d

def return_image_embedding(model,img_path):
    x = tensorimage.img_to_array(img_path)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    curr_df = pd.DataFrame(preds[0]).T
    return curr_df
model = ResNet50(include_top=False, weights='imagenet', pooling='avg')



# Function to add a tick to the image based on similarity score
def add_tick_to_pdf_image(image, bounding_coords, colour_similarity, similarity_score_dinov2):
    draw = ImageDraw.Draw(image)
    if colour_similarity < 30 and similarity_score_dinov2 > 0.70:
        draw.rectangle(bounding_coords,outline='green', width=5)
    else:
        draw.rectangle(bounding_coords,outline='red', width=5)
    return image

def add_tick_to_shelf_image(image, bounding_coords, colour_similarity, similarity_score_dinov2):
    draw = ImageDraw.Draw(image)
    if colour_similarity < 30 and similarity_score_dinov2 > 0.70:
        draw.rectangle(bounding_coords,outline='green', width=5)
    else:
        draw.rectangle(bounding_coords,outline='red', width=5)
    return image

def finding_logo_colour(image, boudingbox_df):
    shelf_bounding_coords = (
                            boudingbox_df['x'][i] + (boudingbox_df['x2'][i] - boudingbox_df['x'][i])*0.1,
                            boudingbox_df['y'][i] + (boudingbox_df['y2'][i] - boudingbox_df['y'][i])*0.1,
                            boudingbox_df['x2'][i] - (boudingbox_df['x2'][i] - boudingbox_df['x'][i])*0.1,
                            boudingbox_df['y2'][i] - (boudingbox_df['y2'][i] - boudingbox_df['y'][i])*0.1) 
    colour_shelf_product = image.crop(shelf_bounding_coords)
    return colour_shelf_product

    
if page == "Final Similarity Check":
    shelf_embeddings_df = pd.DataFrame(columns=['index', 'dino_embeddings', 'res_embeddings'])
    pdf_embeddings_df = pd.DataFrame(columns=['index', 'dino_embeddings', 'res_embeddings'])

    if st.session_state.image_uploaded_file is not None:
        image = Image.open(st.session_state.image_uploaded_file)
        image_array = np.asarray(image)
        uploaded_image = ImageOps.exif_transpose(image)
    else:
        st.write("Please Upload the shelf planogram images as well as pdf")

    if st.session_state.shelf_boundingbox is not None:
        shelf_result = st.session_state.shelf_boundingbox
        shelf_cv_image =st.session_state.shelf_cv_image 
        cv_img_visualized = visualize_recognition_result(shelf_result, shelf_cv_image)
        cv_img_visualized = cv2.cvtColor(cv_img_visualized, cv2.COLOR_BGR2RGB)
        product_detected_image = Image.fromarray(cv_img_visualized) 
        st.image([uploaded_image,product_detected_image], use_column_width=True) 
        shelf_boundingbox = pd.DataFrame(columns=['id', 'x', 'y', 'x2','y2','w', 'h','group'])
        for prediction in shelf_result['products']:
            shelf_details = {
        'id':[prediction['id']],
        'x':[prediction['boundingBox']['x']],
        'y':[prediction['boundingBox']['y']],
        'x2': [prediction['boundingBox']['x'] + prediction['boundingBox']['w']],
        'y2': [prediction['boundingBox']['y'] + prediction['boundingBox']['h']],
        'w' : [prediction['boundingBox']['w']],
        'h' : [prediction['boundingBox']['h']],
        'group' : 'NA'
            }
            shelf_boundingbox = pd.concat([shelf_boundingbox, pd.DataFrame(shelf_details)],ignore_index=True)

        for prediction in shelf_result['gaps']:
            shelf_details = {
        'id':[prediction['id']],
        'x':[prediction['boundingBox']['x']],
        'y':[prediction['boundingBox']['y']],
        'x2': [prediction['boundingBox']['x'] + prediction['boundingBox']['w']],
        'y2': [prediction['boundingBox']['y'] + prediction['boundingBox']['h']],
        'w' : [prediction['boundingBox']['w']],
        'h' : [prediction['boundingBox']['h']],
        'group' : 'NA'
            }
            shelf_boundingbox = pd.concat([shelf_boundingbox, pd.DataFrame(shelf_details)],ignore_index=True)
        

        # Sort bounding boxes from top-left to bottom-right
        shelf_boundingBoxGrouped_df  = pd.DataFrame()
        boundingBoxGrouped_list = list()
        j = 0

        while (not shelf_boundingbox.empty):
            shelf_boundingbox = shelf_boundingbox.sort_values(by=['y2','x2'], ascending= True)
            firstBb = shelf_boundingbox.iloc[0]
            for index, row in shelf_boundingbox.iterrows():
                if(row['y2'] < firstBb['y2'] + firstBb['h']*0.45 and row['y2'] > firstBb['y2'] - 0.45*firstBb['h']):
                    # add to row array row_array
                    row['group'] = j
                    idval = row['id']
                    boundingBoxGrouped_list.append(row)
                    shelf_boundingbox = shelf_boundingbox.drop(index)
            j = j+ 1
        shelf_boundingBoxGrouped_df = pd.DataFrame(boundingBoxGrouped_list)
        shelf_boundingbox = shelf_boundingBoxGrouped_df.sort_values(by=['group', 'x2'], ascending=True) 
        shelf_boundingbox.index = pd.RangeIndex(start=0, stop=len(shelf_boundingbox), step=1)
        shelf_boundingbox = shelf_boundingbox.reset_index()
    
        # Instantiate the embedder and detector outside the loop for efficiency
        embedder = Embedder(architecture="dinov2_", device="cpu")
        
        # Iterate over the sorted bounding boxes len(shelf_boundingbox)
        for i in range(len(shelf_boundingbox)):
            masks = {}
            mask_name_cropped = {}
            shelf_bounding_coords = (
                                    shelf_boundingbox['x'][i],
                                    shelf_boundingbox['y'][i],
                                    shelf_boundingbox['x2'][i],
                                    shelf_boundingbox['y2'][i]) 
            cropped_shelf_product = image.crop(shelf_bounding_coords)
            shelf_image_cropped = transform(cropped_shelf_product)
            shelf_dinov2 = embedder.predict([shelf_image_cropped])
            shelf_res50 = return_image_embedding(model,cropped_shelf_product)
            shelf_embeddings_df = pd.concat([shelf_embeddings_df,pd.DataFrame(
                {'index':shelf_boundingbox['index'][i],
                 'dino_embeddings':shelf_dinov2,
                 'res_embeddings':list(shelf_res50.values)
                 })],ignore_index=True)

 ##############Planogram PDF 
    if st.session_state.cropped_image is not None:
        pdf_image = Image.open(st.session_state.cropped_image)
        pdf_image_array = np.asarray(pdf_image)
        pdf_uploaded_image = ImageOps.exif_transpose(pdf_image)
    else:
        st.write("Please Upload the shelf planogram images as well as pdf")
        # st.write(uploaded_image)
    if st.session_state.pdf_boundingbox is not None:
        pdf_result = st.session_state.pdf_boundingbox
        pdf_cv_image =st.session_state.pdf_cv_image 
        pdf_img_visualized = visualize_recognition_result(pdf_result, pdf_cv_image)
        pdf_img_visualized = cv2.cvtColor(pdf_img_visualized, cv2.COLOR_BGR2RGB)
        pdf_detected_image = Image.fromarray(pdf_img_visualized) 
        st.image([pdf_uploaded_image,pdf_detected_image], use_column_width=True) 
        pdf_boundingbox_df = pd.DataFrame(columns=['id', 'x', 'y', 'x2','y2','w', 'h','group'])
        for prediction in pdf_result['products']:
            pdf_details = {
                            'id':[prediction['id']],
                            'x':[prediction['boundingBox']['x']],
                            'y':[prediction['boundingBox']['y']],
                            'x2': [prediction['boundingBox']['x'] + prediction['boundingBox']['w']],
                            'y2': [prediction['boundingBox']['y'] + prediction['boundingBox']['h']],
                            'w' : [prediction['boundingBox']['w']],
                            'h' : [prediction['boundingBox']['h']],
                            'group' : 'NA'
            }
            pdf_boundingbox_df = pd.concat([pdf_boundingbox_df, pd.DataFrame(pdf_details)],ignore_index=True)
        
        for prediction in pdf_result['gaps']:
            pdf_details = {
                            'id':[prediction['id']],
                            'x':[prediction['boundingBox']['x']],
                            'y':[prediction['boundingBox']['y']],
                            'x2': [prediction['boundingBox']['x'] + prediction['boundingBox']['w']],
                            'y2': [prediction['boundingBox']['y'] + prediction['boundingBox']['h']],
                            'w' : [prediction['boundingBox']['w']],
                            'h' : [prediction['boundingBox']['h']],
                            'group' : 'NA'
                          }
            pdf_boundingbox_df = pd.concat([pdf_boundingbox_df, pd.DataFrame(pdf_details)],ignore_index=True)
        # Sort bounding boxes from top-left to bottom-right
        pdf_boundingBoxGrouped_df  = pd.DataFrame()
        pdf_boundingBoxGrouped_list = list()
        j = 0

        while (not pdf_boundingbox_df.empty):
            pdf_boundingbox_df = pdf_boundingbox_df.sort_values(by=['y2','x2'], ascending= True)
            firstBb = pdf_boundingbox_df.iloc[0]
            for index, row in pdf_boundingbox_df.iterrows():
                if(row['y2'] < firstBb['y2'] + firstBb['h']*0.45 and row['y2'] > firstBb['y2'] - 0.45*firstBb['h']):
                    # add to row array row_array
                    row['group'] = j
                    idval = row['id']
                    pdf_boundingBoxGrouped_list.append(row)
                    pdf_boundingbox_df = pdf_boundingbox_df.drop(index)
            j = j+ 1
        number_of_group = j
        pdf_boundingBoxGrouped_df = pd.DataFrame(pdf_boundingBoxGrouped_list)
        pdf_boundingbox_df = pdf_boundingBoxGrouped_df.sort_values(by=['group', 'x2'], ascending=True) 
        pdf_boundingbox_df.index = pd.RangeIndex(start=0, stop=len(pdf_boundingbox_df), step=1)
        pdf_boundingbox_df = pdf_boundingbox_df.reset_index()
        
        # Instantiate the embedder and detector outside the loop for efficiency
        embedder = Embedder(architecture="dinov2_", device="cpu")
        
        # Iterate over the sorted bounding boxes len(shelf_boundingbox)
        for i in range(len(pdf_boundingbox_df)):
            pdf_masks = {}
            pdf_mask_name_cropped = {}
            pdf_embeddings = {}
            pdf_bounding_coords = (pdf_boundingbox_df['x'][i],
                                    pdf_boundingbox_df['y'][i],
                                    pdf_boundingbox_df['x2'][i],
                                    pdf_boundingbox_df['y2'][i])
            cropped_pdf_product = pdf_image.crop(pdf_bounding_coords)
            pdf_image_cropped = transform(cropped_pdf_product)
            pdf_dinov2 = embedder.predict([pdf_image_cropped])
            pdf_resnet50 = return_image_embedding(model,cropped_pdf_product)
            pdf_embeddings_df = pd.concat([pdf_embeddings_df,pd.DataFrame({'index':pdf_boundingbox_df['index'][i],'dino_embeddings':pdf_dinov2,'res_embeddings':list(pdf_resnet50.values)})],ignore_index=True)

        for k in range(number_of_group):
            shelf_group_df = shelf_boundingbox[shelf_boundingbox['group'] == k]
            shelf_embeddings_group_df = shelf_embeddings_df[shelf_embeddings_df['index'].isin(shelf_group_df['index'])]
            shelf_rows = shelf_group_df.sort_values(by= ['index'], ascending= True)
            shelf_embeddings_group_df = shelf_embeddings_group_df.sort_values(by= ['index'], ascending= True)
            shelf_embeddings_group_df = shelf_embeddings_group_df.reset_index()
            shelf_rows = shelf_rows.reset_index()
            pdf_group_df = pdf_boundingbox_df[pdf_boundingbox_df['group'] == k]
            pdf_embeddings_group_df = pdf_embeddings_df[pdf_embeddings_df['index'].isin(pdf_group_df['index'])]
            pdf_rows = pdf_group_df.sort_values(by= ['index'], ascending= True)
            pdf_rows = pdf_rows.reset_index()
            pdf_embeddings_group_df = pdf_embeddings_group_df.sort_values(by= ['index'], ascending= True)
            pdf_embeddings_group_df = pdf_embeddings_group_df.reset_index()
            length_rows = len(shelf_rows)
            if len(pdf_rows) <= len(shelf_rows):
                length_rows = len(pdf_rows)
            for i in range(length_rows):
                shelf_bounding_coords = (
                                        shelf_rows['x'][i],
                                        shelf_rows['y'][i],
                                        shelf_rows['x2'][i],
                                        shelf_rows['y2'][i]) 
                cropped_shelf_product = image.crop(shelf_bounding_coords)
                shelf_colour = np.array(finding_logo_colour(image,shelf_rows))
                shelf_colour_average = shelf_colour.mean(axis=0).mean(axis=0)

                shelf_color_average_lab = rgb2lab(shelf_colour_average/255)
                pdf_bounding_coords = (pdf_rows['x'][i],
                                    pdf_rows['y'][i],
                                    pdf_rows['x2'][i],
                                    pdf_rows['y2'][i])
                cropped_pdf_product = pdf_image.crop(pdf_bounding_coords)
                similarity_score_dinov2 = cosine_similarity(np.array(shelf_embeddings_group_df['dino_embeddings'][i]).reshape(1,-1),np.array(pdf_embeddings_group_df['dino_embeddings'][i]).reshape(1,-1)).item()
                similarity_score_res50 = cosine_similarity(np.array(shelf_embeddings_group_df['res_embeddings'][i]).reshape(1,-1),np.array(pdf_embeddings_group_df['res_embeddings'][i]).reshape(1,-1)).item()
                pdf_colour = np.array(finding_logo_colour(pdf_image,pdf_rows))
                pdf_color_average = pdf_colour.mean(axis=0).mean(axis=0)
                pdf_color_average_lab = rgb2lab(pdf_color_average/255)
                colour_similarity = deltaE_ciede2000(shelf_color_average_lab, pdf_color_average_lab)
                image = add_tick_to_shelf_image(image, shelf_bounding_coords, colour_similarity,similarity_score_dinov2)
                pdf_image = add_tick_to_pdf_image(pdf_image, pdf_bounding_coords, colour_similarity,similarity_score_dinov2)

    # Display the final images with ticks
        st.write("Green Box Indicates Compliant and Red Box Indicates Non-Compliant with PDF")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Shelf Image with Ticks", use_column_width=True)
        with col2:
            st.image(pdf_image, caption="PDF Image with Ticks", use_column_width=True)