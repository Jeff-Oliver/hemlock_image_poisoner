import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from PIL import Image

DEBUG = False  # Set to True for debugging output

################################################################################
def convert_image_to_tensor(image):
    """
    Convert an image to a TensorFlow tensor if it is not already in that format. Ensure
    the image is in the correct shape for processing. If the image is a batch of images,
    it will be reshaped to remove the batch dimension. If the image is a single image,
    it will be reshaped to add a batch dimension. The function also ensures that the
    image is of type float32 for compatibility with TensorFlow operations.

    Args:
        image (numpy.ndarray, TensorFlow tensor, or tuple): The input image to be converted.
    
    Returns:
        tensor: The converted image as a tensor.
    """

    if DEBUG == True:
        print("Before conversion - Image Type:", type(image))
        print("Before conversion - Image Shape:", image.shape)

    # Ensure image is a TensorFlow tensor with dtype float32
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    # Ensure the input image has the correct shape
    if len(image.shape) == 5:  # Check if the shape is (1, 1, 224, 224, 3)
        image = tf.squeeze(image, axis=0)  # Remove the first dimension
        print("Corrected CW image shape:", image.shape)  # Should be (1, 224, 224, 3)
    elif len(image.shape) == 3:  # Check if the shape is (224, 224, 3)
        image = tf.expand_dims(image, axis=0)  # Add a batch dimension
        print("Corrected CW image shape:", image.shape)  # Should be (1, 224, 224, 3)

    if DEBUG == True:
        print("After conversion - Image Type:", type(image))
        print("After conversion - Image Shape:", image.shape)

    return image

################################################################################
def convert_label_to_tensor(label):
    """
    Convert a label to a TensorFlow tensor if it is not already in that format.
    
    Args:
        label (numpy.ndarray, TensorFlow tensor, or tuple): The input label to be converted.
    
    Returns:
        tensor: The converted label as a tensor.
    """

    if DEBUG == True:
        print("Before conversion - Label Type:", type(label))
        print("Before conversion - Label Shape:", label.shape if hasattr(label, 'shape') else "No shape attribute")

    # Ensure label is a TensorFlow tensor with dtype float32
    if not isinstance(label, tf.Tensor):
        label = tf.convert_to_tensor(label, dtype=tf.float32)

    if DEBUG == True:
        print("After conversion - Label Type:", type(label))
        print("After conversion - Label Shape:", label.shape if hasattr(label, 'shape') else "No shape attribute")

    return label

################################################################################
def convert_label_to_scalar_int(label):
    """
    Convert a label to a scalar integer if it is not already in that format.
    
    Args:
        label (numpy.ndarray, TensorFlow tensor, or tuple): The input label to be converted.
    
    Returns:
        tensor: The converted label as a scalar integer.
    """

    if DEBUG == True:
        print("Before conversion - Label Type:", type(label))
        print("Before conversion - Label Shape:", label.shape if hasattr(label, 'shape') else "No shape attribute")

    # Ensure label is a TensorFlow tensor with dtype float32
    if not isinstance(label, tf.Tensor):
        label = tf.convert_to_tensor(label, dtype=tf.float32)

    # Convert label from shape (1, 1000) to a scalar integer
    if len(label.shape) == 2 and label.shape[0] == 1:
        label = tf.argmax(label, axis=1)  # Get the class index with the highest probability
        if DEBUG == True:
            print("Label value after argmax:", label.numpy())  # Print the class index

    label = tf.squeeze(label)  # Remove any remaining dimensions of size 1
    label = tf.cast(label, tf.int32)  # Ensure label is a scalar integer

    if DEBUG == True:
        print("After conversion - Label Type:", type(label))
        print("After conversion - Label Shape:", label.shape if hasattr(label, 'shape') else "No shape attribute")

    return label

################################################################################
def FGSM(image, label, model, loss_func, eps):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    This method generates adversarial examples by adding a small perturbation to
    the input image in the direction of the gradient of the loss with respect to
    the input image. The perturbation is scaled by a small factor (eps) to ensure
    the modified image remains within a certain distance from the original image.

    Args:
        image (numpy.ndarray, TensorFlow tensor, or tuple): The input image.
        label (numpy.ndarray, TensorFlow tensor, or tuple): The label of the input image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        loss_func (tf.keras.losses): The loss function used to compute the gradients.
        eps (float): The maximum perturbation allowed.

    Returns:
        Adversarial image (NumPy array).
        numpy.ndarray: The signed gradients used to create the perturbation.
    """

    # Debugging: print the types and shape of the image and label
    if DEBUG == True:
        print("=======================================")
        print("=== FGSM image and label conversion ===")
        print("=======================================")        

    image = convert_image_to_tensor(image)  # Convert image to tensor and ensure correct shape
    label = convert_label_to_tensor(label)  # Convert label to tensor and ensure correct shape

    # Calculate the gradient of the loss with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(image) # Watch the input image for gradient calculation
        prediction = model(image) # Get the model's predictions
        loss = loss_func(label, prediction) # Calculate the loss using the specified loss function

    # Get the gradients of the loss with respect to the input image
    gradients = tape.gradient(loss, image)

    # Get the sign of the gradients to create the perturbation
    signed_gradients = tf.sign(gradients)

    # Display the perturbation image
    plt.imshow(signed_gradients[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
    plt.title('FGSM Perturbation')
    plt.axis('off')
    plt.show()

    # Create the adversarial image by adjusting the original image with the signed gradients
    adversarial_image = image + eps * signed_gradients

    # Clip the pixel values to be in the valid range [0, 1]
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    # Debugging: print the shapes of the images and labels
    if DEBUG == True:
        print("FGSM Image shape:", image.shape)
        print("FGSM Prediction shape:", prediction.shape)
        print("FGSM Perturbation shape:", signed_gradients.shape)
        print("FGSM Adversarial image shape:", adversarial_image.numpy().shape)
        print("FGSM Label shape:", label.shape if hasattr(label, 'shape') else "No shape attribute")


    return adversarial_image.numpy(), signed_gradients.numpy()

################################################################################
def PGD(image, label, model, loss_func, eps=0.3, max_iterations=40, alpha=0.01, targeted=False):
    """
    Projected Gradient Descent (PGD) adversarial attack.
    This method iteratively perturbs the input image in the direction of the gradient of the loss
    with respect to the input image, scaled by a small factor (eps).
    The perturbation is added to the original image to create an adversarial example.
    The adversarial image is then clipped to ensure pixel values remain in the valid range [0, 1].
    The PGD method is an iterative version of the FGSM attack, which allows for more control
    over the perturbation and can lead to stronger adversarial examples.

    Args:
        image (numpy.ndarray, TensorFlow tensor, or tuple): The input image.
        label (numpy.ndarray, TensorFlow tensor, or tuple): The label of the input image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        loss_func (tf.keras.losses): The loss function used to compute the gradients.
        eps (float): The maximum perturbation allowed.
        max_iterations (int): The number of iterations for the attack.
        alpha (float): The step size for for gradient ascent/descent.
        targeted (boolean): Whether to perform a targeted attack.

    Returns:
        Adversarial image (NumPy array).
    """

    # Debugging: print the types and shape of the image and label
    if DEBUG == True:
        print("======================================")
        print("=== PGD image and label conversion ===")
        print("======================================")        

    image = convert_image_to_tensor(image)  # Convert image to tensor and ensure correct shape
    label = convert_label_to_tensor(label)  # Convert label to tensor and ensure correct shape

    # Create a copy of the original image to modify
    adversarial_image = tf.identity(image)
    #adversarial_image = tf.Variable(tf.zeros_like(image), dtype=tf.float32)

    # Perform the attack for the specified number of iterations
    for _ in range(int(max_iterations)):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image) # Watch the adversarial image for gradient calculation
            predictions = model(adversarial_image) # Get the model's predictions
            loss = loss_func(label, predictions) # Calculate the loss using the specified loss function

        # Get the gradients of the loss with respect to the adversarial image
        gradients = tape.gradient(loss, adversarial_image)
        
        # Get the sign of the gradients
        signed_gradients = tf.sign(gradients)
        
        if targeted:
            # If the attack is targeted, subtract the normalized gradients
            adversarial_image = adversarial_image - alpha * signed_gradients
        else:
            # If the attack is untargeted, add the normalized gradients
            adversarial_image = adversarial_image + alpha * signed_gradients

        # Clip the pixel values to be in the valid range [0, 1]
        perturbation = tf.clip_by_value(adversarial_image - image, -eps, eps)
        adversarial_image = tf.clip_by_value(image + perturbation, 0, 1)
         
    # Display the perturbation image
    plt.imshow(signed_gradients[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
    plt.title('PGD Perturbation')
    plt.axis('off')
    plt.show()

    # Debugging: print the shapes of the images and labels
    if DEBUG == True:
        print("PGD Image shape:", image.shape)
        print("PGD Prediction shape:", predictions.shape)
        print("PGD Perturbation shape:", signed_gradients.shape)
        print("PGD Adversarial image shape:", adversarial_image.numpy().shape)
        print("PGD Label shape:", label.shape if hasattr(label, 'shape') else "No shape attribute")

    return adversarial_image.numpy()

################################################################################
def CW(image, label, model, targeted=False, c=1, kappa=0, max_iterations=1000, learning_rate=0.01):
    """
    Carlini-Wagner (CW) adversarial attack.
    This method generates adversarial examples using the CW attack.
    
    Args:
        image (numpy.ndarray): The input image.
        label (numpy.ndarray): The labels of the input image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        targeted: (boolean): Whether to perform a targeted attack.
        c (float): Weight for the L2 regularization term.
        kappa (float): Confidence parameter.
        max_iterations (int): Maximum number of iterations for optimization.
        learning_rate (float): Learning rate for optimization.
              
    Returns:
        Adversarial image (NumPy array).
    """

    # define the loss function for the CW attack
    def loss_fn(x_adv):
        outputs = model(x_adv) # Get the model's predictions
        real = tf.reduce_sum(outputs * label, axis=1) # Isolate the logit for the true label

        # Get the logit for the other classes to identify the most confident misclassification
        other = tf.reduce_max((1 - label) * outputs - (label * 10000), axis=1)

        # Return the loss based on whether the attack is targeted or not
        if targeted:
            return tf.maximum(other - real, -kappa)
        else:
            return tf.maximum(real - other, -kappa)

    # Debugging: print the types and shape of the image and label
    if DEBUG == True:
        print("=======================================")
        print("=== CW image and label conversion ===")
        print("=======================================")        
    
    image = convert_image_to_tensor(image)  # Convert image to tensor and ensure correct shape
    label = convert_label_to_tensor(label)  # Convert label to tensor and ensure correct shape
    
    # Initialize the adversarial image as a trainable variable
    w = tf.Variable(tf.zeros_like(image), dtype=tf.float32)

    # Define the optimizer
    w_opt = tf.keras.optimizers.Adam(learning_rate)

    best_l2 = float('inf') # Initialize best L2 distance with infinity
    best_adv = image # Initialize best adversarial image with the original image

    # Perform the optimization process
    for iteration in range(max_iterations):
        with tf.GradientTape() as tape:
            adv_images = tf.tanh(w) * 0.5 + 0.5 # Transform w to the range [0, 1]

           # Compute the Perturbation loss which is the L2 distance between the adversarial image and the original image
            l2_dist = tf.reduce_sum(tf.square(adv_images - image), axis=[1, 2, 3])

            # Calculate the misclassification loss
            loss1 = c * loss_fn(adv_images)

            # Assign the L2 distance (Perturbation loss) to the loss2 variable
            loss2 = l2_dist

            # Compute the total loss
            loss = tf.reduce_sum(loss1 + loss2)

        gradients = tape.gradient(loss, w) # Get the gradients of the loss with respect to w
        w_opt.apply_gradients([(gradients, w)]) # Apply the gradients to update w

        # Print the loss every 10% of the iterations
        if iteration % (max_iterations // 10) == 0:
            print(f"Iteration: {iteration}, Loss: {loss.numpy()}")

        # Update the adversarial image and best L2 distance if the loss is less than or equal to 0
        # and the L2 distance is less than the best L2 distance
        if tf.reduce_max(loss_fn(adv_images)) <= 0 and tf.reduce_mean(l2_dist) < best_l2:
            best_l2 = tf.reduce_mean(l2_dist)
            best_adv = adv_images

    return best_adv.numpy()

################################################################################
# Future Work: Implement the JSMA attack
################################################################################
def JSMA(image, label, model, theta=1, max_iterations=50):
    """
    Jacobian-based Saliency Map Attack (JSMA) adversarial attack.
    This method uses the saliency map of the input image to identify the most
    important pixels to modify in order to mislead the model. The attack is
    performed by iteratively modifying the pixels with the highest saliency
    until the model misclassifies the image or a maximum number of modifications
    is reached. The JSMA attack is known for its ability to create visually
    imperceptible adversarial examples.
    
    Args:
        image (numpy.ndarray, TensorFlow tensor, or tuple): The input image.
        label (numpy.ndarray, TensorFlow tensor, or tuple): The label of the input image.
        model (tf.keras.Model): The model used for generating adversarial examples.
        theta (float): The maximum perturbation allowed.
        max_iterations (int): The maximum number of iterations.
    
    Returns:
        Adversarial image (NumPy array).
    """

    # Debugging: print the types and shape of the image and label
    if DEBUG == True:
        print("=======================================")
        print("=== JSMA image and label conversion ===")
        print("=======================================")        
    
    image_tf = convert_image_to_tensor(image)  # Convert image to tensor and ensure correct shape
    label_tf = tf.convert_to_tensor([label], dtype=tf.int32) # Convert label to tensor
    #label_tf = convert_label_to_scalar_int(label) # Convert label to scalar integer

    def saliency_map(image_var, label_var):
        with tf.GradientTape() as tape:
            tape.watch(image_var)
            outputs = model(image_var)
            target_output = outputs[0, label_var[0]]

        gradients = tape.gradient(target_output, image_var)[0]
        gradients_flat = tf.reshape(gradients, [-1])
        saliency = tf.abs(gradients_flat)
        return saliency.numpy()

    def forward_derivative(image_var):
        with tf.GradientTape() as tape:
            tape.watch(image_var)
            outputs = model(image_var)
        jacobian = tape.jacobian(outputs, image_var)[0, :, 0, :, :, 0]
        return jacobian.numpy()

    adv_image = tf.identity(image_tf)  # Create a copy of the original image
    adv_image_tf = tf.convert_to_tensor(adv_image, dtype=tf.float32)

    for _ in range(max_iterations):
        saliency = saliency_map(adv_image_tf, label_tf)
        jacobian = forward_derivative(adv_image_tf)

        # Find the two pixels with the highest saliency
        saliency_flat = saliency.flatten()
        top_indices = np.argpartition(saliency_flat, -2)[-2:]

        # Calculate the forward derivative for the target class
        jacobian_target = jacobian[label]
        jacobian_target_flat = jacobian_target.flatten()

        # Check if the Jacobian elements are positive
        if jacobian_target_flat[top_indices[0]] > 0 and jacobian_target_flat[top_indices[1]] > 0:
            # Perturb the pixels
            adv_image_flat = adv_image.flatten()
            adv_image_flat[top_indices[0]] += theta
            adv_image_flat[top_indices[1]] += theta
            adv_image = adv_image_flat.reshape(image.shape)
            adv_image = np.clip(adv_image, 0, 1) #ensure image remains in valid range.
            adv_image_tf = tf.convert_to_tensor(adv_image, dtype=tf.float32)

            # Check if the target class is predicted
            prediction = np.argmax(model.predict(adv_image), axis=1)[0]
            if prediction == label:
                return adv_image

        else:
            return image #attack failed.

    return adv_image #return best adversarial image found.

################################################################################
# Future Work: Implement the DeepFool attack
################################################################################
def DeepFool(image, num_classes, model, max_iterations=50, overshoot=0.02):
    """
    DeepFool adversarial attack.
    This method iteratively perturbs the input image in the direction of the
    nearest decision boundary of the model. The perturbation is added to the
    original image to create an adversarial example. The DeepFool attack is
    known for its effectiveness against various models and is often used as a
    benchmark for adversarial robustness. The attack is performed by iteratively
    calculating the perturbation needed to cross the decision boundary and
    updating the adversarial image accordingly. The process continues until
    the model misclassifies the image or a maximum number of iterations is reached.

    Args:
        image: Input image (NumPy array, shape (1, height, width, channels)).
        num_classes: Number of classes in the model.
        model: A TensorFlow Keras model.
        max_iterations: Maximum number of iterations.
        overshoot: Overshoot parameter.

    Returns:
        Adversarial image (NumPy array).
    """

    image_tf = tf.convert_to_tensor(image, dtype=tf.float32)

    original_label = np.argmax(model.predict(image), axis=1)[0]
    original_output = model(image_tf)[0]

    adversarial_image = tf.identity(image_tf)
    adversarial_image_tf = tf.convert_to_tensor(adversarial_image, dtype=tf.float32)

    i = 0
    current_label = original_label

    while current_label == original_label and i < max_iterations:
        with tf.GradientTape() as tape:
            tape.watch(adversarial_image_tf)
            outputs = model(adversarial_image_tf)[0]

        gradients = tape.gradient(outputs, adversarial_image_tf)

        # Compute the adversarial perturbation
        w = np.inf
        best_l = None
        for k in range(num_classes):
            if k == current_label:
                continue
            wk = gradients[0] - gradients[0] #initialize wk to zero.
            fk = outputs[k] - outputs[current_label]

            wk = gradients[0] - tf.gather(gradients[0], k, axis =0) #gradients[0] is the gradient for the current label.
            fk = outputs[k] - outputs[current_label]

            wk_norm = tf.norm(tf.reshape(wk, [-1]))
            if wk_norm == 0:
              wk_norm = 1e-8 #prevent division by zero.
            lk = abs(fk) / wk_norm
            if lk < w:
                w = lk
                best_l = wk

        r_i = (w + overshoot) * best_l / tf.norm(tf.reshape(best_l, [-1]))
        adversarial_image = adversarial_image + r_i.numpy()
        adversarial_image = np.clip(adversarial_image, 0, 1) #ensure image remains in the valid range.
        adversarial_image_tf = tf.convert_to_tensor(adversarial_image, dtype=tf.float32)

        current_label = np.argmax(model.predict(adversarial_image), axis=1)[0]
        i += 1

    return adversarial_image

################################################################################
def visualize_adversarial_examples(original_image, adversarial_image, labels, model, description):
    """
    Visualize the original and adversarial images along with their predictions.
    This function uses Matplotlib to create a side-by-side comparison of the original
    and adversarial images, displaying the predicted labels for both.

    Args:
        original_image (numpy.ndarray): The original image.
        adversarial_image (numpy.ndarray): The adversarial image.
        labels (callable): A function to decode model predictions into human-readable labels.
        model (tf.keras.Model): The model used for generating predictions.
    """

    # Ensure the images are numpy arrays
    if not isinstance(original_image, np.ndarray):
        original_image = original_image.numpy()
    if not isinstance(adversarial_image, np.ndarray):
        adversarial_image = adversarial_image.numpy()

    # Get the predicted labels for both images
    original_prediction = model.predict(original_image)
    adversarial_prediction = model.predict(adversarial_image)

    # Decode predictions into human-readable labels
    original_label = labels(original_prediction, top=1)[0][0]
    adversarial_label = labels(adversarial_prediction, top=1)[0][0]

    # Normalize images to [0, 1] range for display
    original_image = (original_image * 0.5) + 0.5  # Convert from [-1, 1] to [0, 1]
    adversarial_image = (adversarial_image * 0.5) + 0.5  # Convert from [-1, 1] to [0, 1]

    if DEBUG == True:
        print("Original image shape:", original_image.shape)
        #print("Original Prediction:", original_prediction)
        print(f'Original Label: {original_label}')
        print("Adversarial image shape:", adversarial_image.shape)
        #print("Adversarial Prediction:", adversarial_prediction)
        print(f'Adversarial Label: {adversarial_label}')

    # Create a figure to display the images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display the original image
    axes[0].imshow(original_image.squeeze())  # No need for additional normalization
    axes[0].set_title(f'Original Image\nLabel: {original_label[1]} ({original_label[2]*100:.2f}%)')
    axes[0].axis('off')

    # Display the adversarial image
    axes[1].imshow(adversarial_image.squeeze())  # No need for additional normalization
    axes[1].set_title(f'{description} Adversarial Image\nLabel: {adversarial_label[1]} ({adversarial_label[2]*100:.2f}%)')
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

    print("============================================")
    print("=== display_dtype_in_tuple function call ===")
    print("============================================")   
   

    # Display the type and dtype of each element in the tuple
    for i, element in enumerate(arg):
        print(f"Element {i}: Type = {type(element)}", end="")
        if hasattr(element, "dtype"):  # Check if the element has a dtype attribute
            print(f", dtype = {element.dtype}")
        else:
            print(", No dtype attribute")
    print(f"Tuple: {type(arg)}")
    # print("\n")

################################################################################
def debug_image_output(image):
    """
    Debugging function to visualize the output of the adversarial attacks.
    This function is designed to be called in a Jupyter notebook or similar
    environment where interactive plotting is supported.

    Args:
        image: The input image.

    Returns:
        None
    """

    print("\n========================================")
    print("=== debug_image_output function call ===")
    print("========================================")   

    # Display the type and dtype of the image
    print("\nImage type and dtypes:")
    if isinstance(image, tuple):
        display_dtype_in_tuple(image)
    else:
        print("Image type:", type(image)) # Print the type of the image
        print("Image dtypes:", image.dtype if hasattr(image, 'dtype') else "No dtype attribute") # Print the dtype of the image

    print("\nOther Image Information:")
    print("Image shape:", image.shape if hasattr(image, 'shape') else "No shape attribute") # Print the shape of the image
    print("Image pixel values:", image.numpy().flatten()[:10]) # print first 10 pixel values

################################################################################
def debug_label_output(label):
    """
    Debugging function to visualize the output of the adversarial attacks.
    This function is designed to be called in a Jupyter notebook or similar
    environment where interactive plotting is supported.

    Args:
        label: The label of the image.

    Returns:
        None
    """

    print("\n========================================")
    print("=== debug_label_output function call ===")
    print("========================================")   

    # Display the type and dtype of the label
    print("\nLabel type and dtypes:")
    if isinstance(label, tuple):
        display_dtype_in_tuple(label)
    else:
        print("Label type:", type(label)) # Print the type of the label
        print("Label dtypes:", label.dtype if hasattr(label, 'dtype') else "No dtype attribute")

    print("\nOther Label Information:")
    print("Label shape:", label.shape if hasattr(label, 'shape') else "No shape attribute") # Print the shape of the label
    print("Label:", label) # Print the label
    if isinstance(label, tuple):
        if len(label) == 3:
            print("Label ID:", label[0]) # Print the label ID
            print("Label name:", label[1]) # Print the label name
            print("Label confidence:", label[2]) # Print the label probability
        else:
            print("Label confidence:", label[0]) # Print the label probability
    else:
        print("Label ID:", label[0])
