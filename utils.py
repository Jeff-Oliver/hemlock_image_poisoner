
#!pip install rembg
#!pip install onnxruntime # required for rembg
#!pip install tensorflow-hub

# import dependencies
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from rembg import remove
import numpy as np
import io
import tensorflow_hub as hub

# ################################################################################
# # Helper function to preprocess the image so that it can be inputted in MobileNetV2
# def preprocess(image):
#   image = tf.cast(image, tf.float32)
#   image = tf.image.resize(image, (224, 224), method=tf.image.ResizeMethod.AREA)
#   image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
#   image = image[None, ...]
#   return image

# ################################################################################
# # Helper function to extract labels from probability vector
# def get_imagenet_label(probs):
#   return decode_predictions(probs, top=1)[0][0]

################################################################################
# Helper function to remove the background from an image tensor
# 
def remove_background(image, background_color=(255, 255, 255)):
    """
    Removes the background from a TensorFlow image tensor, crops to a square bounding box,
    and replaces the background with a specified color.

    Args:
        image (tf.Tensor): Input image tensor.
        background_color (tuple): RGB color for the new background (default is white).

    Returns:
        tf.Tensor: Image tensor with the background removed, cropped, and replaced with the specified color.
    """
    image_np = image.numpy()  # Convert TensorFlow tensor to NumPy array
    pil_image = Image.fromarray(image_np)

    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format="PNG")  # Save the PIL image to a bytes buffer
    input_image = img_bytes.getvalue()  # Get the image bytes
    output_image = remove(input_image)  # Remove the background using rembg

    # Convert the output bytes back to a PIL image
    pil_image_no_bg = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # Crop to the bounding box
    bbox = pil_image_no_bg.getchannel("A").getbbox() # Get the bounding box of the alpha channel
    if bbox:
        pil_image_no_bg = pil_image_no_bg.crop(bbox)
    else:
        raise ValueError("No bounding box found. The image might be fully transparent.")

    # Determine the size of the square canvas (longest side of the bounding box)
    width, height = pil_image_no_bg.size
    square_size = max(width, height)

    # Create a new square image with the specified background color
    background = Image.new("RGBA", (square_size, square_size), background_color + (255,))

    # Calculate the position to paste the cropped image (centered)
    paste_position = ((square_size - width) // 2,
                      (square_size - height) // 2)

    # Paste the cropped image onto the new background
    background.paste(pil_image_no_bg, paste_position, pil_image_no_bg.getchannel("A"))

    # Convert the image to RGB (to remove transparency)
    combined_image = background.convert("RGB")

    # Convert the PIL image back to a TensorFlow tensor
    image = tf.convert_to_tensor(np.array(combined_image), dtype=tf.uint8)

    return image
################################################################################
# Helper function to identify the image subject and crop to a square bounding box
# Load a pre-trained object detection model from TensorFlow Hub
def detect_and_crop(image):
    """
    Detects objects in the image and crops to a square bounding box around the largest detected object.

    Args:
        image (tf.Tensor): Input image tensor.

    Returns:
        PIL.Image: Cropped image around the largest detected object with a square bounding box.
    """
    # Convert TensorFlow tensor to NumPy array
    image_np = image.numpy()

    # Add batch dimension and normalize the image
    input_tensor = tf.convert_to_tensor(image_np[None, ...], dtype=tf.uint8)

    # Run the object detection model
    detections = detector(input_tensor)

    # Extract bounding boxes and scores
    boxes = detections["detection_boxes"][0].numpy()  # Normalized bounding boxes
    scores = detections["detection_scores"][0].numpy()

    # Select the box with the highest score
    max_score_idx = np.argmax(scores)
    box = boxes[max_score_idx]

    # Convert normalized box coordinates to pixel coordinates
    height, width, _ = image_np.shape
    ymin, xmin, ymax, xmax = box
    (left, top, right, bottom) = (int(xmin * width), int(ymin * height),
                                  int(xmax * width), int(ymax * height))

    # Calculate the center of the bounding box
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    # Calculate the size of the square (longest side of the bounding box)
    box_width = right - left
    box_height = bottom - top
    square_size = max(box_width, box_height)

    # Adjust the bounding box to make it square
    half_size = square_size // 2
    left = max(0, center_x - half_size)
    right = min(width, center_x + half_size)
    top = max(0, center_y - half_size)
    bottom = min(height, center_y + half_size)

    # Crop the image to the square bounding box
    pil_image = Image.fromarray(image_np)
    cropped_image = pil_image.crop((left, top, right, bottom))

    return cropped_image

################################################################################
# Helper function to crop the image to a square
def crop_to_square(image):
    """
    Crops an image to a square by removing excess on the longer side.

    Args:
        image (tf.Tensor or PIL.Image.Image): Input image to be cropped.

    Returns:
        PIL.Image.Image: The cropped square image.
    """
    # Ensure the image is a Pillow Image
    if isinstance(image, tf.Tensor):
        image = Image.fromarray(image.numpy())  # Convert TensorFlow tensor to Pillow Image

    # Get the dimensions of the image
    width, height = image.size
    shortest_side = min(width, height)

    # Calculate cropping box
    left = (width - shortest_side) // 2
    top = (height - shortest_side) // 2
    right = (width + shortest_side) // 2
    bottom = (height + shortest_side) // 2

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

################################################################################
# Helper function to add white space to the image that makes it square
def expand_to_square(image, background_color=(255, 255, 255)):
    """
    Expands an image to a square by adding whitespace (padding) to the shorter side.

    Args:
        image (PIL.Image.Image): Input image to be expanded.
        background_color (tuple): RGB color for the padding (default is white).

    Returns:
        PIL.Image.Image: The square image with added whitespace.
    """
    # Ensure the image is a Pillow Image
    if isinstance(image, tf.Tensor):
        image = Image.fromarray(image.numpy())  # Convert TensorFlow tensor to Pillow Image

    # Get the dimensions of the image
    width, height = image.size

    # Determine the size of the square canvas (longest side of the image)
    square_size = max(width, height)

    # Create a new square image with the specified background color
    square_image = Image.new("RGB", (square_size, square_size), background_color)

    # Calculate the position to paste the original image (centered)
    paste_position = ((square_size - width) // 2, (square_size - height) // 2)

    # Paste the original image onto the square canvas
    square_image.paste(image, paste_position)

    return square_image