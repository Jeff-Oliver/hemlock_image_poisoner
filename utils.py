from PIL import Image
from rembg import remove



################################################################################
def crop_to_square(input_path, output_path):
    """
    Crops an image to a square by removing excess on the longer side
    and saves the result to the specified output path.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the cropped image.
    """
    # Open the input image
    with Image.open(input_path) as img:
        # Get the dimensions of the image
        width, height = img.size
        shortest_side = min(width, height)

        # Calculate cropping box
        left = (width - shortest_side) // 2
        top = (height - shortest_side) // 2
        right = (width + shortest_side) // 2
        bottom = (height + shortest_side) // 2

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Save the cropped image to the output path
        cropped_img.save(output_path)



################################################################################
def make_square_image(image_path, output_path):
    """
    Adds whitespace to an image and makes it square.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the modified image.
    """
    img = Image.open(image_path)
    width, height = img.size

    # Determine the larger dimension
    max_dim = max(width, height)

    # Create a new image with a white background and square dimensions
    new_img = Image.new("RGB", (max_dim, max_dim), "white")

    # Calculate the position to paste the original image
    x_offset = (max_dim - width) // 2
    y_offset = (max_dim - height) // 2

    # Paste the original image onto the new image
    new_img.paste(img, (x_offset, y_offset))

    # Save the modified image
    new_img.save(output_path)

################################################################################
# this import required for background removal
# from rembg import remove

def remove_background(input_path, output_path):
    """
    Removes the background from an image and saves the result.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the output image with the background removed.
    """
    with open(input_path, 'rb') as img:
        input_image = img.read()
        output_image = remove(input_image)

    with open(output_path, 'wb') as o:
        o.write(output_image)



################################################################################
def crop_to_bounding_box(input_path, output_path):
    """
    Crops an image to its bounding box, centers it in a square canvas with a black background,
    and saves the result as a JPEG.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the cropped image.
    """
    # Open the input image
    img = Image.open(input_path)
    # Convert the image to RGBA if not already in that mode
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    # Get the bounding box of the alpha channel
    bbox = img.getchannel("A").getbbox()

    if bbox:
        # Extract the bounding box coordinates
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top

        # Determine the size of the square (longest side)
        square_size = max(width, height)

        # Calculate padding to center the image
        horizontal_padding = (square_size - width) // 2
        vertical_padding = (square_size - height) // 2

        # Adjust the bounding box to make it square
        square_bbox = (
            left,
            top,
            right,
            bottom
        )

        # Crop the image to the bounding box
        cropped_img = img.crop(square_bbox)
        
        # Create a new square image with a black background
        black_background = Image.new("RGBA", (square_size, square_size), (0, 0, 0, 255))

        # Calculate the position to paste the cropped image (centered)
        paste_position = (horizontal_padding, vertical_padding)

        # Paste the cropped image onto the black background using the alpha channel as a mask
        black_background.paste(cropped_img, paste_position, cropped_img.getchannel("A"))

        # Convert the image to RGB (JPEG does not support transparency)
        black_background = black_background.convert("RGB")

        # Save the result as a JPEG
        black_background.save(output_path, format="JPEG")
        print(f"Image cropped, centered, and saved to {output_path}")
    else:
        print("No bounding box found. The image might be fully transparent.")
    



