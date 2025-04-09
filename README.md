# Team 5 - Hemlock Image Poisoner

## Project Contributors
- Saman Zahra Raza
- Mark Murphy
- Leonardo Rastelli Galebe
- Steven Frasica
- Jeff Oliver 
- Dennis Kipng'eno Bett

## Overview
In this project, our team developed Hemlock: A Poison Image Creator and Detector. Hemlock is an advanced system designed to apply and identify adversarial perturbations using the Fast Gradient Sign Method (FGSM). We implemented an intuitive user interface that enables users to automatically generate poisoned images, thereby embedding protective signals into their visual content. In the current digital age, social media users face increasing threats of content theft and unauthorized use of their media in machine learning models. This tool offers an additional security layer by enabling content creators—particularly influencers and digital artists—to embed adversarial noise as a form of copyright protection. Our model serves as a proactive defense against content scraping, bot-generated impersonation, and deepfake misuse, while also supporting digital content ownership and traceability across platforms.

## MobileNetV2 Model 
MobileNetV2 is an existing deep learning CNN model for image classification. 

## Advesirial Attack Methods

1. Fast Gradient Sign Method: Perturbs input data by moving it slightly in the direction of the gradient sign to maximize model error.

2. Projected Gradient Descent Method: Applies small FGSM steps repeatedly and projects the adversarial example back into a valid range after each step.

3. Carlini & Wagner Approach: Minimizes perturbations while still causing misclassification, often seen as one of the most effective attacks against deep networks.

## Installation
### **1. Clone the Repository**
```bash
git clone (https://github.com/Jeff-Oliver/hemlock_image_poisoner.git

cd hemlock_image_poisoner
```

### **2. Install Dependencies**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn statsmodels streamlit joblib
```

### **3. Run the Streamlit App**
```bash
streamlit run interface.py
```

### **4. Verify Installation**
Run the following command to check if Streamlit is installed correctly:
```bash
streamlit --version
```

**Packages:**

#!pip install rembg

#!pip install onnxruntime

#!pip install tensorflow-hub

## Program Usage Guide
1. **Run `main.ipynb` first** to ensure all necessary data is processed correctly before using the recommendation system.
2. **Launch the Streamlit Web App**.
3. **Use the sidebar sliders** to set song characteristics (e.g., danceability, energy, tempo) depending on your preference.
4. Click **"Find My Song"** to get songs recommendation.
5. The system will return **up to 10 similar songs** based on the selected algorithm.

## Example Output
Below is an example of the expected Streamlit UI when a user searches for songs:

**Screeshot**

## File Structure
```
                       Hemlock_Image_Poisoner

│── images                             # images used to train model

│── gitignore                          # Contains pre-trained ML models & processed datasets

│── README.md                          # Project Documentation (This File)

│── adversarial_attacks.py             # Adversarial Attack Functions

│── main.ipynb                         # Jupyter Notebook for data exploration & modeling (MUST be run first)

│── utils.py                           # Utility functions for EDA & preprocessing

│── presentation                       # Powerpoint presentation ()pdf format for project showcase
```


## Troubleshooting

**Missing File Errors**

- Solution: Make sure you have run `main.ipynb` before launching `interface.py`. This generates all the necessary datasets required to run the program.

## Powerpoint Presentation
 **Powerpoint Presentation**: Available on [Google Slides](https://docs.google.com/presentation/d/1bXuiae8r6g7LAxt3JwkmMEAAOndjQzjU-EggEg0uJOE/edit?usp=sharing")

 ## Conclusion & Next Steps

 In order to maintain a reasonable scope for this project we had to keep our initial work very brief. For this initial phase we worked on cleaning up the model & trying out various adversial attack methods. For our next steps if we ever continue this project we will take some time on developing a robust Streamlit interface. A lot of our code is commented out within our code. 
 
 
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
