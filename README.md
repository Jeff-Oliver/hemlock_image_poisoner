# Team 5 - Hemlock Image Poisoner

## Project Contributors
- Saman Zahra Raza
- Mark Murphy
- Leonardo Rastelli Galebe
- Steven Frasica
- Jeff Oliver 
- Dennis Kipng'eno Bett

## Overview
In this project, our team is developing Hemlock. Hemlock is an advanced system designed to apply various attack methods to pertubate images. This will allow for a proactive defense against content scraping, bot-generated impersonation, and deepfake misuse, while also supporting digital content ownership and traceability across platforms. In this project we have succesfully completed Phase 1. In this initial phase we have succesfully used MobileNetV2 Image Classification and applied our attack methods to the images so that the model missclassifies them. 

## MobileNetV2 Model
For this project, we used MobileNetV2, an existing deep learning convolutional neural network (CNN) model for Image classification, Object detection, and Semantic segmentation. This model derives its images from image net. We decided on this model due to its efficiency with limited computation, lightweight architecture, and ease to integrate into attack pipelines as the victim model.

## Adversarial Attack Methods

These are the attack methods that our Model currently supports:

1. Fast Gradient Sign Method: Perturbs input data by moving it slightly in the direction of the gradient sign to maximize model error. This attack method is convenient for evaluating vulnerability of the model.

2. Projected Gradient Descent Method: Applies small FGSM steps repeatedly and projects the adversarial example back into a valid range after each step. This attack method helps simulate worst-case attacks under a fixed perturbation budget.

3. Carlini & Wagner Approach: Minimizes perturbations while still causing misclassification, often seen as one of the most effective attacks against deep networks. This is especially due to its stealthness with the atttacks being very hard to detect.

## Installing and running the program

In order to get our mdoel to work please complete the following steps: 
### **1. Clone the Repository**
```bash
git clone (https://github.com/Jeff-Oliver/hemlock_image_poisoner.git

cd hemlock_image_poisoner
```

### **2. Install Dependencies**
```bash
pip install tensorflow tensorflow_hub matplotlib pathlib numpy pandas PIL rembg io onnxruntime scikit-image
```

### **3. Run the main.ipynb file**

## File Structure
```
                       Hemlock_Image_Poisoner

│── images                             # images used to train model

│── gitignore                          # Contains pre-trained ML models & processed datasets

│── README.md                          # Project Documentation (This File)

│── adversarial_attacks.py             # Adversarial Attack Functions

│── main.ipynb                         # Jupyter Notebook for data exploration & modeling (MUST be run first)

│── utils.py                           # Utility functions for EDA & preprocessing

│── presentation                       # Powerpoint presentation ()pdf format for project showcase. Also includes demo video. 
```


## Troubleshooting
In case you come across any issues with the program, please: 

(1) Make sure you have run `main.ipynb` first

(2) Ensure your packages are succesfully installed as demonstrated (above). 

## Powerpoint Presentation
 **Powerpoint Presentation**: Available on [Google Slides](https://docs.google.com/presentation/d/1bXuiae8r6g7LAxt3JwkmMEAAOndjQzjU-EggEg0uJOE/edit?usp=sharing")

 ## Conclusion & Next Steps 

For Phase 2 we will do the following: Add more loss functions, experiment with other image attack methods, calculate a true Attack Success Rate, and create a robust User Interface which allows users to select their attack methods etc. We will also scale out our model so it can work on Video and Audio media. Since we had to de-scope this project a lot of our future state work has been commented out. 
 
 
## Acknowledgments
This project utilizes the following Model & Datasets
1. **MobileNetV2 Model**: Available on [Keras]( https://keras.io/api/applications/mobilenet/#mobilenetv2-function), "This model returns a Keras image classification model, optionally loaded with weights pre-trained on ImageNet"
2. **Model Documentation**: Available on [Keras]( https://keras.io/api/applications/#usage-examples-for-image-classification-models)
3. **Tutorial Consulted**:  Available on [Tensorflow](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)
4. **Co-Lab Documentation Consulted**:  Available on [Co-Lab](https://colab.research.google.com/drive/1bOzVaDQo8h6Ngstb7AcfzC35OihpHspt)
  
## Contribution Guideline
Contributions are welcome and appreciated! Please follow these steps:
1. **Fork** the repository.
2. Create a **feature branch** (`git checkout -b feature-branch`).
3. **Commit your changes** (`git commit -m "Added new feature"`).
4. **Push to GitHub** (`git push origin feature-branch`).
5. Open a **Pull Request**.

## License
This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
