import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image

################################################################################
def convert_image_to_Tensor(image):
    """
    Convert an image to a NumPy array if it is not already in that format.
    This function is useful for ensuring that the image data is in a format
    that can be processed by various machine learning frameworks.

    Args:
        image (numpy.ndarray, TensorFlow tensor, or tuple): The input image to be converted.
    
    Returns:
        numpy.ndarray: The converted image as a NumPy array.
    """
    
    # Check if the image is already a NumPy array
    # if isinstance(image, np.ndarray):
    #     return image
    # # If the image is a TensorFlow tensor, convert it to a NumPy array
    # elif isinstance(image, tf.Tensor):
    #     image = image.numpy()
    # # If the image is a tuple, convert it to a NumPy array
    # elif isinstance(image, tuple):
    #     image = np.array(image)

    # Ensure image is a TensorFlow tensor with dtype float32
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    # If model expects a batch dimension, reshape the image
    # if len(image.shape) == 3:
    #     image = np.expand_dims(image, axis=0)


    return image

################################################################################
def convert_label_to_Tensor(label):
    """
    Convert a label to a NumPy array if it is not already in that format.
    This function is useful for ensuring that the label data is in a format
    that can be processed by various machine learning frameworks.
    
    Args:
        label (numpy.ndarray, TensorFlow tensor, or tuple): The input label to be converted.
    
    Returns:
        numpy.ndarray: The converted label as a NumPy array.
    """

    # Check if the label is already a NumPy array
    # if isinstance(label, np.ndarray):
    #     return label
    # # If the label is a TensorFlow tensor, convert it to a NumPy array
    # elif isinstance(label, tf.Tensor):
    #     label = label.numpy()
    # # If the label is a tuple, convert it to a NumPy array
    # elif isinstance(label, tuple):
    #     label = np.array(label)

    # # If the label is a string, convert it to an integer
    # if not isinstance(label[0], str):
    #     label[0] = str(label[0])
    
    # if not isinstance(label[1], str):
    #     label[1] = str(label[1])

    # if not isinstance(label[2], float):
    #     label[2] = float(label[2])

    # Ensure label is a TensorFlow tensor with dtype float32
    if not isinstance(label, tf.Tensor):
        label = tf.convert_to_tensor(label, dtype=tf.float32)

    # Ensure label is a TensorFlow tensor with dtype int64
    # if not isinstance(label, tf.Tensor):
    #     label = tf.convert_to_tensor(label, dtype=tf.int64)

    # If model expects a batch dimension, reshape the image
    # if len(label.shape) == 3:
    #     label = np.expand_dims(label, axis=0)

    return label

################################################################################
def FGSM(image, label, model, loss_func, eps):
    """
    Generate adversarial image using the Fast Gradient Sign Method (FGSM).
    This method perturbs the input image in the direction of the gradient of the loss
    with respect to the input image, scaled by a small factor (eps).
    The perturbation is added to the original image to create an adversarial example.
    The adversarial image is then clipped to ensure pixel values remain in the valid range [0, 1].
    This function is designed to work with TensorFlow and Keras models.

    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The true label of the image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        eps (float): The maximum perturbation allowed.

    Returns:
        numpy.ndarray: The adversarial image.
    """

    image = convert_image_to_Tensor(image)

    # if "SparseCategoricalCrossentropy" in loss:
    #     label = (label[-1],)

    label = convert_label_to_Tensor(label)
    
    debug_output(image, label)

    # Calculate the gradient of the loss with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)

        print("Label shape:", label.shape)
        print("Prediction shape:", prediction.shape)
        
        # # Ensure label is 1D
        # if len(label.shape) > 1:
        #     label = tf.squeeze(label)
        #if "SparseCategoricalCrossentropy" in loss:
        loss = loss_func(label, prediction)
        # elif "CategoricalCrossentropy" in loss:
        #     loss = loss(label, prediction)


    # Get the gradients of the loss with respect to the input image
    gradients = tape.gradient(loss, image)

    # Get the sign of the gradients
    signed_gradients = tf.sign(gradients)

    # Create the adversarial image by adjusting the original image with the signed gradients
    adversarial_image = image + eps * signed_gradients

    # Clip the pixel values to be in the valid range [0, 1]
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    print("Image shape:", image.shape)
    print("FGSM image shape:", adversarial_image.numpy().shape)
    print("Label shape:", label.shape)

    return adversarial_image.numpy()

################################################################################
def PGD(image, label, model, eps=0.1, num_iterations=40):
    """
    Generate adversarial image using the Projected Gradient Descent (PGD) method.
    This method iteratively perturbs the input image in the direction of the gradient of the loss
    with respect to the input image, scaled by a small factor (eps).
    The perturbation is added to the original image to create an adversarial example.
    The adversarial image is then clipped to ensure pixel values remain in the valid range [0, 1].
    This function is designed to work with TensorFlow and Keras models.
    The PGD method is an iterative version of the FGSM attack, which allows for more control
    over the perturbation and can lead to stronger adversarial examples.

    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The true label of the image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        eps (float): The maximum perturbation allowed.
        num_iterations (int): The number of iterations for the attack.

    Returns:
        numpy.ndarray: The adversarial image.
    """
    
    # Ensure the image is a tensor
    image = convert_image_to_Tensor(image)

    # Ensure the label is a tensor
    label = convert_label_to_Tensor(label)

    # Create a copy of the original image to modify
    adversarial_image = tf.identity(image)

    # Perform the attack for the specified number of iterations
    for _ in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            prediction = model(adversarial_image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)

        # Get the gradients of the loss with respect to the adversarial image
        gradients = tape.gradient(loss, adversarial_image)

        # Get the sign of the gradients
        signed_gradients = tf.sign(gradients)

        # Update the adversarial image
        adversarial_image = adversarial_image + eps * signed_gradients

        # Clip the pixel values to be in the valid range [0, 1]
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    return adversarial_image.numpy()

################################################################################
def CW(image, label, model, confidence=0, max_iter=1000, learning_rate=0.01, c=1e-4):
    """
    Generate adversarial image using the Carlini & Wagner (CW)
    attack method. This method is a more sophisticated attack that optimizes
    the perturbation to minimize the difference between the original and
    adversarial images while ensuring the adversarial image is misclassified.
    The CW attack is known for its effectiveness against various models and
    is often used as a benchmark for adversarial robustness.

    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The true label of the image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        confidence (float): Confidence margin for misclassification.
        max_iter (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for optimization.
        c (float): Regularization parameter for the loss function.

    Returns:
        numpy.ndarray: The adversarial image.
    """

    # Ensure the image is a tensor
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Initialize variables
    adversarial_image = tf.Variable(image)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define the loss function
    def loss_fn():
        logits = model(adversarial_image[None, ...])
        real = logits[0, label]
        other = tf.reduce_max(tf.concat([logits[0, :label], logits[0, label + 1:]], axis=0))
        # Misclassification loss
        misclassification_loss = tf.maximum(0.0, other - real + confidence)
        # Regularization loss
        perturbation_loss = tf.reduce_sum(tf.square(adversarial_image - image))
        return c * misclassification_loss + perturbation_loss

    # Perform optimization
    for _ in range(max_iter):
        optimizer.minimize(loss_fn, var_list=[adversarial_image])
        # Clip the adversarial image to ensure pixel values are in the valid range [0, 1]
        adversarial_image.assign(tf.clip_by_value(adversarial_image, 0, 1))

    return adversarial_image.numpy()

################################################################################
def JSMA(image, label, model, eps=0.1, max_iter=100):
    """
    Generate adversarial image using the Jacobian-based Saliency Map Attack (JSMA).
    This method uses the saliency map of the input image to identify the most
    important pixels to modify in order to mislead the model. The attack is
    performed by iteratively modifying the pixels with the highest saliency
    until the model misclassifies the image or a maximum number of modifications
    is reached. The JSMA attack is known for its ability to create visually
    imperceptible adversarial examples.
    
    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The true label of the image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        eps (float): The maximum perturbation allowed.
        max_iter (int): The maximum number of pixel modifications.
    
    Returns:
        numpy.ndarray: The adversarial image.
    """

    # Ensure the image is a tensor
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Initialize the adversarial image
    adversarial_image = tf.identity(image)

    for _ in range(max_iter):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            logits = model(adversarial_image[None, ...])
        
        # Compute the Jacobian (gradients of logits w.r.t. the input image)
        gradients = tape.gradient(logits, adversarial_image)

        # Compute the saliency map for the target label
        target_grad = gradients[0, label]
        other_grad = tf.reduce_sum(gradients[0, :label], axis=0) + tf.reduce_sum(gradients[0, label + 1:], axis=0)
        saliency_map = tf.abs(target_grad) - tf.abs(other_grad)

        # Find the pixel with the highest saliency
        max_pixel = tf.argmax(tf.reshape(saliency_map, [-1]))
        max_pixel_coords = tf.unravel_index(max_pixel, saliency_map.shape)

        # Modify the pixel in the direction of the gradient
        perturbation = tf.sign(target_grad[max_pixel_coords]) * eps
        adversarial_image = tf.tensor_scatter_nd_add(adversarial_image, [max_pixel_coords], [perturbation])

        # Clip the pixel values to be in the valid range [0, 1]
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

        # Check if the model misclassifies the image
        prediction = tf.argmax(model(adversarial_image[None, ...]), axis=1).numpy()[0]
        if prediction != label:
            break

    return adversarial_image.numpy()

################################################################################
def DeepFool(image, label, model, max_iter=50, eps=1e-6):
    """
    Generate adversarial image using the DeepFool attack method.
    This method iteratively perturbs the input image in the direction of the
    nearest decision boundary of the model. The perturbation is added to the
    original image to create an adversarial example. The DeepFool attack is
    known for its effectiveness against various models and is often used as a
    benchmark for adversarial robustness. The attack is performed by iteratively
    calculating the perturbation needed to cross the decision boundary and
    updating the adversarial image accordingly. The process continues until
    the model misclassifies the image or a maximum number of iterations is reached.
    
    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The true label of the image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        eps (float): The maximum perturbation allowed.
    
    Returns:
        numpy.ndarray: The adversarial image.
    """

    # Ensure the image is a tensor
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Get the original prediction
    original_prediction = tf.argmax(model(image[None, ...]), axis=1).numpy()[0]

    # Initialize variables
    adversarial_image = tf.identity(image)
    perturbation = tf.zeros_like(image)
    iteration = 0

    while original_prediction == label and iteration < max_iter:
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image)
            logits = model(adversarial_image[None, ...])
            loss = logits[0, label]

        # Compute gradients of the loss with respect to the image
        gradients = tape.gradient(loss, adversarial_image)

        # Compute the perturbation
        w = gradients
        f = logits[0, label]
        perturbation_step = tf.abs(f) / (tf.norm(w) + eps) * tf.sign(w)

        # Update the adversarial image
        perturbation += perturbation_step
        adversarial_image = tf.clip_by_value(image + perturbation, 0, 1)

        # Update the prediction
        original_prediction = tf.argmax(model(adversarial_image[None, ...]), axis=1).numpy()[0]
        iteration += 1

    return adversarial_image.numpy()
    

################################################################################
def visualize_adversarial_examples(original_image, adversarial_image, labels, model):
    """
    Visualize the original and adversarial images along with their predictions.
    This function uses Matplotlib to create a side-by-side comparison of the original
    and adversarial images, displaying the predicted labels for both.

    Args:
        original_image (numpy.ndarray): The original image.
        adversarial_image (numpy.ndarray): The adversarial image.
        label (numpy.ndarray): The true label of the image.
        model (tf.keras.Model): The model used for generating predictions.
    """

    # Ensure the images are numpy arrays
    if not isinstance(original_image, np.ndarray):
        original_image = original_image.numpy()
    if not isinstance(adversarial_image, np.ndarray):
        adversarial_image = adversarial_image.numpy()

    
    # Convert the label to a NumPy array if it is not already
    print("Image shape:", original_image.shape)
    print("FGSM image shape:", adversarial_image.shape)


    # Get the predicted labels for both images
    #original_prediction = model.predict(np.expand_dims(original_image, axis=0))
    original_prediction = model.predict(original_image)
    #adversarial_prediction = model.predict(np.expand_dims(adversarial_image, axis=0))
    adversarial_prediction = model.predict(adversarial_image)

    # Convert predictions to class labels
    #original_label = np.argmax(original_prediction)
    original_label = labels(original_prediction, top=1)[0][0]
    #adversarial_label = np.argmax(adversarial_prediction)
    adversarial_label = labels(adversarial_prediction, top=1)[0][0]

    debug_output(original_image, original_label)
    debug_output(adversarial_image, adversarial_label)

    print(f'Original Label: {original_label}')
    print(f'Adversarial Label: {adversarial_label}')

    # Create a figure to display the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    #axes[0].imshow(original_image.squeeze(), cmap='gray')
    axes[0].imshow(original_image.squeeze())
    #axes[0].set_title(f'Original Image\nTrue Label: {label}\nPredicted: {original_label}')
    axes[0].set_title('Original Image\nTrue Label: {} : {:.2f}% Confidence'.format(original_label[1], original_label[2]*100))
    axes[0].axis('off')

    # Display the adversarial image
    #axes[1].imshow(adversarial_image.squeeze(), cmap='gray')
    axes[1].imshow(adversarial_image.squeeze())
    #axes[1].set_title(f'Adversarial Image\nTrue Label: {label}\nPredicted: {adversarial_label}')
    axes[1].set_title('Adversarial Image\nTrue Label: {} : {:.2f}% Confidence'.format(adversarial_label[1], adversarial_label[2]*100))
    axes[1].axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()

################################################################################
def save_adversarial_images(original_image, adversarial_image, label, output_dir):
    """
    Save the original and adversarial images to the specified directory.
    The images are saved in PNG format with appropriate filenames.

    Args:
        original_image (numpy.ndarray): The original image.
        adversarial_image (numpy.ndarray): The adversarial image.
        label (numpy.ndarray): The true label of the image.
        output_dir (str): The directory where the images will be saved.
    """

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create filenames for the images
    original_filename = Path(output_dir) / f'original_label_{label[1]}.png'
    adversarial_filename = Path(output_dir) / f'adversarial_label_{label[1]}.png'

    # Normalize the images to [0, 1] range
    original_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))
    adversarial_image = (adversarial_image - np.min(adversarial_image)) / (np.max(adversarial_image) - np.min(adversarial_image))
    
    # Save the images
    mpl.image.imsave(original_filename, original_image)
    mpl.image.imsave(adversarial_filename, adversarial_image)
    print(f'Saved original image to {original_filename}')
    print(f'Saved adversarial image to {adversarial_filename}')

#################################################################################
def display_dtype_in_tuple(arg):
    """
    Display the type and dtype of each element in a tuple.
    This function is useful for debugging purposes to understand the
    data types of the elements in a tuple.

    Args:
        arg (tuple): The input tuple to be inspected.

    Returns:
        None
    """

    # Display the type and dtype of each element in the tuple
    for i, element in enumerate(arg):
        print(f"Element {i}: Type = {type(element)}", end="")
        if hasattr(element, "dtype"):  # Check if the element has a dtype attribute
            print(f", dtype = {element.dtype}")
        else:
            print(", No dtype attribute")
    print(f"Tuple: {type(arg)}")
    print("\n")

################################################################################
def debug_output(image, label):
    """
    Debugging function to visualize the output of the adversarial attacks.
    This function is designed to be called in a Jupyter notebook or similar
    environment where interactive plotting is supported.

    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The true label of the image.

    Returns:
        None
    """

    print(f"Image: {type(image)}")
    print(image.shape if hasattr(image, 'shape') else "No shape attribute")
    print(image.dtype if hasattr(image, 'dtype') else "No dtype attribute")
    display_dtype_in_tuple(image)


    print(f"Label: {type(label)}")
    print(label.shape if hasattr(label, 'shape') else "No shape attribute")
    print(label)
    #display_dtype_in_tuple(label)
    
################################################################################
# def main():
#     # Load your model and data here
#     model = tf.keras.applications.MobileNetV2(weights='imagenet',
#                                           include_top=True) # Load the model with ImageNet weights
#     model.trainable = False # Freeze the model

#     # ImageNet labels
#     label = tf.keras.applications.mobilenet_v2.decode_predictions

#     image = get_image()  # Load your image
#     # image_probs = model.predict(image) # returns a probability vector for the likelyhood of each class
    
#     # loss_object = tf.keras.losses.CategoricalCrossentropy() # Loss function to be used for the adversarial attack

#     # labrador_retriever_index = 208 # Index of the label in the ImageNet dataset
#     # label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1]) # One-hot encoding of the label
#     # label = tf.reshape(label, (1, image_probs.shape[-1])) # Reshape the label to match the input shape of the model

#     # perturbations = create_adversarial_pattern(image, label) # Create the perturbations using the adversarial pattern function
#     # plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]

#     # Generate adversarial images
#     fgsm_image = FGSM(image, label, model)
#     pgd_image = PGD(image, label, model)
#     cw_image = CW(image, label, model)
#     jsma_image = JSMA(image, label, model)
#     deepfool_image = DeepFool(image, label, model)

#     # Visualize the adverarial examples
#     visualize_adversarial_examples(image, fgsm_image, label, model)
#     visualize_adversarial_examples(image, pgd_image, label, model)
#     visualize_adversarial_examples(image, cw_image, label, model)
#     visualize_adversarial_examples(image, jsma_image, label, model)
#     visualize_adversarial_examples(image, deepfool_image, label, model)

#     # Save the original image
#     save_adversarial_images(image, image, label, 'output/original')

#     # Save the adversarial images
#     save_adversarial_images(image, fgsm_image, label, 'output/fgsm')
#     save_adversarial_images(image, pgd_image, label, 'output/pgd')
#     save_adversarial_images(image, cw_image, label, 'output/cw')
#     save_adversarial_images(image, jsma_image, label, 'output/jsma')
#     save_adversarial_images(image, deepfool_image, label, 'output/deepfool')
    

# if __name__ == '__main__':
#     main()

################################################################################