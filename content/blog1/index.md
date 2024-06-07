---
title: "FSAMT : Face Shape Adaptive Makeup Transfer"
description: "Blog posts"
type: "section"
---
***P.S. To protect code ownership and copyright, the code in this article may be simplified, hidden, and frequently use sample data, but this does not hinder the understanding of the core ideas.***
### Summary
This work introduces a “3-level” adaptive makeup transfer framework, addressing facial makeup through two sub-tasks:

1. Makeup adaptation, utilizing feature descrip- tors and eyelid curve algorithms to classify 135 organ-level face shapes;
2. Makeup transfer, achieved by learning the reference picture from three branches (color, highlight, pattern) and applying it to the source picture.
The proposed framework, termed “Face Shape Adaptive Makeup Transfer” (𝐹𝑆𝐴𝑀𝑇), demonstrates superior results in makeup transfer output quality, as confirmed by experimental results.

![Face Shape Adaptive Makeup Transfer (FSAMT) Pipeline](/blogs1/fig1.jpg "Face Shape Adaptive Makeup Transfer (FSAMT) Pipeline")
### Makeup Adaptation Model
### 1. Face Shape Classification
![Makeup Adaptation Model](/blogs1/fig3.jpg "Makeup Adaptation Model")
Overview:

- **Uses PRNet to extract 3D coordinates of the face**: The PRNet (Position Map Regression Network) is utilized to obtain the 3D coordinates of facial landmarks. This provides a detailed geometric representation of the face, which is crucial for accurate shape classification.
- **Classifies face shapes using the SEDNet neural network**: The extracted 3D coordinates are then fed into SEDNet (Squeeze-and-Excitation DenseNet) for face shape classification. SEDNet enhances feature extraction by focusing on important features while suppressing less useful ones.
- **Five types of face shape classifications**: The framework classifies faces into five distinct shapes, which allows for more personalized and accurate makeup adaptation.
#### Code Analysis
##### Initialization and Loading of PRNet Model
```python
import numpy as np
import cv2
from utils.api import PRN

class FaceShapeClassifier:
    def __init__(self, prn_model_path):
        # Initialize the PRNet model
        self.prn = PRN(is_dlib=True)
        self.prn.load_model(prn_model_path)

# Instantiate the classifier
classifier = FaceShapeClassifier('path_to_prn_model')
```
**Explanation**:

- This code initializes the FaceShapeClassifier class and loads the PRNet model using the specified model path. The PRN class is used to handle the 3D face processing tasks.
##### Extraction of Facial Feature Points
```python
class FaceShapeClassifier:
    def __init__(self, prn_model_path):
        self.prn = PRN(is_dlib=True)
        self.prn.load_model(prn_model_path)

    def extract_features(self, face_image):
        # Resize the face image to the required input size for PRNet
        resized_face = cv2.resize(face_image, (256, 256))
        # Process the face image to get 3D position map
        pos = self.prn.process(resized_face)
        # Extract vertices (3D coordinates)
        vertices = self.prn.get_vertices(pos)
        return vertices

# Example usage
face_image = cv2.imread('path_to_face_image.jpg')
features = classifier.extract_features(face_image)
print(features.shape)  # Should print the shape of the extracted features
```
**Explanation**:

- The extract_features method resizes the input face image to 256x256 pixels, which is the required input size for PRNet.
- The PRNet model processes the resized face image to obtain a 3D position map.
- The vertices (3D coordinates) are extracted from the position map and returned for further processing.
##### Training and Testing the SEDNet Model for Face Shape Classification
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEDNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=5):
        super(SEDNet, self).__init__()
        # Dense layer Di
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)

        # SE layer Si
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 64)

        # Classification layer
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Dense layer Di
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # SE layer Si
        b, c, _, _ = x.size()
        y = F.avg_pool2d(x, x.size(2)).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        x = x * y.expand_as(x)

        # Classification layer
        x = F.adaptive_avg_pool2d(x, 1).view(b, -1)
        x = self.fc3(x)
        return x

# Example usage
face_image = cv2.imread('path_to_face_image.jpg')
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
face_image = cv2.resize(face_image, (256, 256)).astype(np.float32) / 255.0
face_image = torch.from_numpy(face_image.transpose(2, 0, 1)).unsqueeze(0).cuda()

# Initialize the model
sednet_model = SEDNet().cuda()
output = sednet_model(face_image)
_, predicted_shape = torch.max(output, 1)
print(f"Predicted face shape: {predicted_shape.item()}")
```
![SED layer of SEDNet](/blogs1/fig4.png "SED layer of SEDNet")
**Explanation**:

Explanation:

- The SEDNet class defines the model architecture based on the provided diagram.
- The Dense layer Di consists of two convolutional layers with ReLU activations and batch normalization.
- The SE layer Si includes global average pooling, followed by two fully connected layers with ReLU and Sigmoid activations, respectively.
- The final classification layer outputs the predicted face shape.
- The example usage demonstrates how to preprocess a face image and use the SEDNet model to predict the face shape.
- This implementation aligns with the structure provided in the diagram and completes the missing parts of the SEDNet model.

This structure provides a comprehensive understanding and implementation of the face shape classification process, from feature extraction using PRNet to classification using SEDNet.
### 2. Eye Shape Classification
![Sampling points](/blogs1/fig5.jpg "Sampling points")![Adjust roundness in long eye](/blogs1/fig6.png "Adjust roundness in long eye")![Adjust roundness in short eye](/blogs1/fig7.png "Adjust roundness in short eye")
#### Overview

- **Classifies eye shapes based on eyelid feature points**: The classification of eye shapes is based on the extraction of key feature points around the eyelids, which provide crucial information about the shape and characteristics of the eyes.
- **Identifies 27 different eye shapes**: The framework identifies and classifies 27 unique eye shapes by combining various parameters like eye length, roundness, and tilt.
#### Code Analysis
##### Extraction of Eye Feature Points
```python
import cv2
import numpy as np

class EyeShapeClassifier(FaceShapeClassifier):
    def extract_eye_features(self, face_image):
        # Extract overall facial features first
        vertices = self.extract_features(face_image)

        # Define key points indices for the eyes (example indices, need actual indices)
        left_eye_indices = [36, 37, 38, 39, 40, 41]
        right_eye_indices = [42, 43, 44, 45, 46, 47]

        # Extract eye feature points
        left_eye_points = vertices[left_eye_indices]
        right_eye_points = vertices[right_eye_indices]

        return left_eye_points, right_eye_points

# Example usage
face_image = cv2.imread('path_to_face_image.jpg')
left_eye_points, right_eye_points = classifier.extract_eye_features(face_image)
print(f"Left Eye Points: {left_eye_points}")
print(f"Right Eye Points: {right_eye_points}")
```
**Explanation**:

- The EyeShapeClassifier class extends FaceShapeClassifier to include methods for extracting eye feature points.
- The extract_eye_features method extracts key feature points for the left and right eyes from the overall facial feature points.
##### Mathematical Fitting of Eyelid Contour Curves
```python
import numpy as np
from scipy.optimize import curve_fit

class EyeShapeClassifier(FaceShapeClassifier):
    def extract_eye_features(self, face_image):
        vertices = self.extract_features(face_image)
        left_eye_indices = [36, 37, 38, 39, 40, 41]
        right_eye_indices = [42, 43, 44, 45, 46, 47]
        left_eye_points = vertices[left_eye_indices]
        right_eye_points = vertices[right_eye_indices]
        return left_eye_points, right_eye_points

    def fit_eyelid_contour(self, eye_points):
        # Define a quadratic function for fitting
        def quadratic_curve(x, a, b, c):
            return a * x**2 + b * x + c

        # Fit upper and lower eyelid curves separately
        upper_eyelid_points = eye_points[:3]  # Example, actual indices may vary
        lower_eyelid_points = eye_points[3:]

        # Fit curves using curve_fit
        upper_params, _ = curve_fit(quadratic_curve, upper_eyelid_points[:, 0], upper_eyelid_points[:, 1])
        lower_params, _ = curve_fit(quadratic_curve, lower_eyelid_points[:, 0], lower_eyelid_points[:, 1])

        return upper_params, lower_params

# Example usage
left_eye_points, right_eye_points = classifier.extract_eye_features(face_image)
left_upper_params, left_lower_params = classifier.fit_eyelid_contour(left_eye_points)
print(f"Left Eye Upper Eyelid Parameters: {left_upper_params}")
print(f"Left Eye Lower Eyelid Parameters: {left_lower_params}")
```
**Explanation**:

- The fit_eyelid_contour method fits quadratic curves to the upper and lower eyelid points using the curve_fit function from the scipy.optimize module.
- This mathematical fitting provides parameters that describe the shape of the eyelid contours.
##### Classification and Labeling of Eye Shapes
```python
class EyeShapeClassifier(FaceShapeClassifier):
    def classify_eye_shape(self, eye_points):
        upper_params, lower_params = self.fit_eyelid_contour(eye_points)

        # Example classification criteria (actual criteria need to be defined)
        eye_length = np.linalg.norm(eye_points[0] - eye_points[3])
        eye_height = max(upper_params[2] - lower_params[2], 0)
        roundness = eye_height / eye_length
        tilt = np.arctan2(eye_points[5][1] - eye_points[0][1], eye_points[5][0] - eye_points[0][0])

        # Classification logic (this is a placeholder, real logic will be based on actual criteria)
        if eye_length < 0.3:
            length_class = 'short'
        elif eye_length < 0.6:
            length_class = 'average'
        else:
            length_class = 'long'

        if roundness < 0.3:
            roundness_class = 'oval'
        else:
            roundness_class = 'round'

        if tilt < -0.1:
            tilt_class = 'drooping'
        elif tilt < 0.1:
            tilt_class = 'horizontal'
        else:
            tilt_class = 'upward'

        eye_shape = (length_class, roundness_class, tilt_class)
        return eye_shape

# Example usage
left_eye_points, right_eye_points = classifier.extract_eye_features(face_image)
left_eye_shape = classifier.classify_eye_shape(left_eye_points)
print(f"Left Eye Shape: {left_eye_shape}")
```
**Explanation**:

- The classify_eye_shape method uses the fitted eyelid contour parameters to derive key attributes such as eye length, height, roundness, and tilt.
- Based on these attributes, it classifies the eye shape into predefined categories.
- This example classification logic is a placeholder, and the actual criteria need to be defined based on the specific requirements and dataset.

This structure provides a comprehensive framework for eye shape classification, from extracting feature points to fitting contour curves and classifying eye shapes based on various attributes.
### 3. Fitting the Eyelid Contour Curve
#### Overview

- **Uses the least squares method to fit eyelid contour curves**: The least squares method is employed to fit a smooth curve to the eyelid contour points, which helps in accurately modeling the shape of the eyelid.
- **Obtains more precise eyelid shapes through fitted curves**: By fitting these curves, more precise and consistent representations of the eyelid shapes are obtained, which are crucial for accurate eye shape classification.
#### Code Analysis
##### Generation of Eyelid Sampling Points
```python
import numpy as np

class EyeShapeClassifier(FaceShapeClassifier):
    def extract_eye_features(self, face_image):
        vertices = self.extract_features(face_image)
        left_eye_indices = [36, 37, 38, 39, 40, 41]
        right_eye_indices = [42, 43, 44, 45, 46, 47]
        left_eye_points = vertices[left_eye_indices]
        right_eye_points = vertices[right_eye_indices]
        return left_eye_points, right_eye_points

    def generate_eyelid_sampling_points(self, eye_points):
        # Assuming eye_points are ordered from the inner corner to the outer corner
        upper_eyelid_points = eye_points[:3]  # Example, actual indices may vary
        lower_eyelid_points = eye_points[3:]

        return upper_eyelid_points, lower_eyelid_points

# Example usage
face_image = cv2.imread('path_to_face_image.jpg')
left_eye_points, right_eye_points = classifier.extract_eye_features(face_image)
left_upper_points, left_lower_points = classifier.generate_eyelid_sampling_points(left_eye_points)
print(f"Left Upper Eyelid Points: {left_upper_points}")
print(f"Left Lower Eyelid Points: {left_lower_points}")
```
**Explanation**:

- The generate_eyelid_sampling_points method separates the eyelid points into upper and lower eyelid points. This allows for fitting separate curves to the upper and lower eyelids.
##### Fitting Eyelid Contour Curves
```python
import numpy as np
from scipy.optimize import curve_fit

class EyeShapeClassifier(FaceShapeClassifier):
    def generate_eyelid_sampling_points(self, eye_points):
        upper_eyelid_points = eye_points[:3]  # Example, actual indices may vary
        lower_eyelid_points = eye_points[3:]
        return upper_eyelid_points, lower_eyelid_points

    def fit_eyelid_contour(self, eye_points):
        def quadratic_curve(x, a, b, c):
            return a * x**2 + b * x + c

        upper_eyelid_points, lower_eyelid_points = self.generate_eyelid_sampling_points(eye_points)

        # Fit upper eyelid curve
        upper_x = upper_eyelid_points[:, 0]
        upper_y = upper_eyelid_points[:, 1]
        upper_params, _ = curve_fit(quadratic_curve, upper_x, upper_y)

        # Fit lower eyelid curve
        lower_x = lower_eyelid_points[:, 0]
        lower_y = lower_eyelid_points[:, 1]
        lower_params, _ = curve_fit(quadratic_curve, lower_x, lower_y)

        return upper_params, lower_params

# Example usage
left_eye_points, right_eye_points = classifier.extract_eye_features(face_image)
left_upper_params, left_lower_params = classifier.fit_eyelid_contour(left_eye_points)
print(f"Left Eye Upper Eyelid Parameters: {left_upper_params}")
print(f"Left Eye Lower Eyelid Parameters: {left_lower_params}")
```
**Explanation**:

- The fit_eyelid_contour method fits quadratic curves to the upper and lower eyelid points using the least squares method (curve_fit function).
- It first generates the sampling points for the upper and lower eyelids.
- Then, it fits the quadratic curve to these points and returns the parameters of the fitted curves.

This structure provides a detailed approach to fitting eyelid contour curves, from generating sampling points to using the least squares method for fitting quadratic curves. This results in more precise representations of the eyelid shapes, which are crucial for accurate eye shape classification.
### 4. Combining Face and Eye Shape Classifications
#### Overview

- **Combines face and eye shape classifications to create 135 organ-level classifications**: By integrating the face shape and eye shape classifications, the framework creates a comprehensive classification system with 135 unique organ-level classifications. This granularity allows for more personalized and accurate makeup adaptation.
- **Provides a more detailed makeup adaptation model**: This detailed classification enhances the makeup adaptation model, allowing it to cater to specific facial features and shapes, resulting in more precise and aesthetically pleasing makeup applications.
#### Code Analysis
##### Combining Results of Face and Eye Shape Classifications
```python
class FaceAndEyeShapeClassifier(EyeShapeClassifier):
    def combine_classifications(self, face_image):
        # Classify face shape
        face_shape = self.classify_face_shape(face_image)

        # Extract eye features
        left_eye_points, right_eye_points = self.extract_eye_features(face_image)

        # Classify eye shapes
        left_eye_shape = self.classify_eye_shape(left_eye_points)
        right_eye_shape = self.classify_eye_shape(right_eye_points)

        # Combine face and eye shape classifications
        combined_classification = {
            "face_shape": face_shape,
            "left_eye_shape": left_eye_shape,
            "right_eye_shape": right_eye_shape
        }

        return combined_classification

# Example usage
face_image = cv2.imread('path_to_face_image.jpg')
combined_classification = classifier.combine_classifications(face_image)
print(f"Combined Classification: {combined_classification}")
```
**Explanation**:

- The combine_classifications method combines the results of face shape and eye shape classifications.
- It first classifies the face shape and then extracts and classifies the left and right eye shapes.
- The combined classification is stored in a dictionary, which provides a comprehensive organ-level classification.
##### Calculating and Storing Classification Results
```python
import json

class FaceAndEyeShapeClassifier(EyeShapeClassifier):
    def combine_classifications(self, face_image):
        face_shape = self.classify_face_shape(face_image)
        left_eye_points, right_eye_points = self.extract_eye_features(face_image)
        left_eye_shape = self.classify_eye_shape(left_eye_points)
        right_eye_shape = self.classify_eye_shape(right_eye_points)

        combined_classification = {
            "face_shape": face_shape,
            "left_eye_shape": left_eye_shape,
            "right_eye_shape": right_eye_shape
        }

        return combined_classification

    def save_classification_results(self, face_image, output_path):
        combined_classification = self.combine_classifications(face_image)

        # Save classification results to a JSON file
        with open(output_path, 'w') as outfile:
            json.dump(combined_classification, outfile, indent=4)

# Example usage
face_image = cv2.imread('path_to_face_image.jpg')
output_path = 'classification_results.json'
classifier.save_classification_results(face_image, output_path)
print(f"Classification results saved to {output_path}")
```
**Explanation**:

- The save_classification_results method combines the face and eye shape classifications and then saves the results to a JSON file.
- This allows for easy storage and retrieval of detailed classification results, which can be used for further analysis or makeup adaptation processes.

This structure combines face and eye shape classifications to create detailed organ-level classifications and provides methods for calculating and storing these results. This comprehensive approach enhances the makeup adaptation model, enabling it to cater to specific facial features and shapes for more personalized makeup applications.
### Makeup Transfer Model
![Makeup Transfer Architecture](/blogs1/fig9.jpg "Makeup Transfer Architecture")
### 1. Color Transfer
#### Overview

- **Proposes a new color transfer method**: The color transfer method involves transferring makeup color from a reference image to a target image, ensuring that the color adapts seamlessly to the target face.
- **Improves GAN architecture with an adaptive binning method**: The GAN architecture is enhanced with an adaptive binning method, which improves the accuracy of color matching and reduces artifacts during the transfer process.
#### Code Analysis
##### Loading and Initializing the Color Transfer Network
```python
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
from utils.models import Generator_branch

class ColorTransfer:
    def __init__(self, checkpoint_path):
        # Initialize the color transfer network
        self.color_net = Generator_branch(64, 6).cuda()
        # Load the pre-trained model weights
        self.color_net.load_state_dict(torch.load(checkpoint_path))
        self.color_net.eval()

    def preprocess_image(self, image_path):
        # Preprocess the image for the model
        image = cv2.imread(image_path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        image = transform(image)
        image = image.unsqueeze(0).cuda()
        return image

# Example usage
color_transfer = ColorTransfer('path_to_color_checkpoint')
image_tensor = color_transfer.preprocess_image('path_to_image.jpg')
print(f"Preprocessed Image Tensor Shape: {image_tensor.shape}")
```
**Explanation**:

- The ColorTransfer class initializes the color transfer network using a pre-trained model checkpoint.
- The preprocess_image method preprocesses the input image to the required format for the color transfer network.
##### Using GAN for Color Transfer
```python
class ColorTransfer:
    def __init__(self, checkpoint_path):
        self.color_net = Generator_branch(64, 6).cuda()
        self.color_net.load_state_dict(torch.load(checkpoint_path))
        self.color_net.eval()

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        image = transform(image)
        image = image.unsqueeze(0).cuda()
        return image

    def transfer_color(self, source_image_path, reference_image_path):
        # Preprocess source and reference images
        source_image = self.preprocess_image(source_image_path)
        reference_image = self.preprocess_image(reference_image_path)

        # Perform color transfer using the GAN
        with torch.no_grad():
            transferred_image = self.color_net(source_image, reference_image)[0]

        # Post-process the transferred image to convert it back to image format
        transferred_image = transferred_image.cpu().numpy()
        transferred_image = np.transpose(transferred_image, (1, 2, 0))
        transferred_image = (transferred_image * 0.5 + 0.5) * 255
        transferred_image = transferred_image.astype(np.uint8)

        return transferred_image

# Example usage
color_transfer = ColorTransfer('path_to_color_checkpoint')
transferred_image = color_transfer.transfer_color('path_to_source_image.jpg', 'path_to_reference_image.jpg')
cv2.imwrite('transferred_image.jpg', transferred_image)
print("Color transfer completed and saved as 'transferred_image.jpg'")
```
**Explanation**:

- The transfer_color method performs the color transfer from the reference image to the source image using the GAN.
- It preprocesses both the source and reference images, passes them through the color transfer network, and post-processes the resulting image to save it in the correct format.
##### Additional Code for Generating Texture
```python
import os
import glob
from tqdm import tqdm
import cv2

class TextureGenerator:
    def __init__(self):
        self.prn = PRN(is_dlib=True)

    def get_texture(self, image, seg):
        pos = self.prn.process(image)
        face_texture = cv2.remap(
            image,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        seg_texture = cv2.remap(
            seg,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        return face_texture, seg_texture

def generate_texture_for_dataset(dataset_path, save_path):
    generator = TextureGenerator()
    list_imgs = glob.glob(os.path.join(dataset_path, "images", "*", "*.png"))
    filenames = [x.split("/all/images/")[-1] for x in list_imgs]
    list_segs = [os.path.join(dataset_path, "segs", x) for x in filenames]

    for i in tqdm(range(0, len(list_imgs))):
        image = cv2.imread(list_imgs[i])
        seg = cv2.imread(list_segs[i])
        uv_texture, uv_seg = generator.get_texture(image, seg)

        subdirs = filenames[i].split('/')
        save_texture_path = os.path.normpath(os.path.join(save_path, "images", *subdirs))
        save_seg_path = os.path.normpath(os.path.join(save_path, "segs", *subdirs))

        os.makedirs(os.path.dirname(save_texture_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_seg_path), exist_ok=True)

        cv2.imwrite(save_texture_path, uv_texture)
        cv2.imwrite(save_seg_path, uv_seg)

# Example usage
generate_texture_for_dataset('./MakeupTransferDataset/all', './MakeupTransfer_UV/')
print("Texture generation completed for the dataset.")
```
**Explanation**:

- The TextureGenerator class generates UV textures for images and their corresponding segmentation masks.
- The generate_texture_for_dataset function processes all images in the specified dataset folder and saves the generated textures and segmentation masks to the specified save directory.

This code provides a comprehensive framework for color transfer using GAN, including the generation of textures for the dataset. It covers the loading and initialization of the color transfer network, performing the color transfer, and generating textures for the dataset, ensuring a complete workflow for the color transfer process.
### 2. Highlight Transfer
![Overview of Highlight Transfer](/blogs1/fig10.png "Overview of Highlight Transfer")
#### Overview

- **Creates the Highlight Face Dataset (HFD)**: The HFD dataset consists of images annotated with highlight regions, enabling the training of models to capture and transfer highlight effects accurately.
- **Uses U-Net for capturing highlight effects**: The U-Net architecture is utilized to detect and transfer highlight effects from reference images to source images, ensuring realistic and aesthetically pleasing results.
#### Code Analysis
##### Generating and Loading the HFD Dataset
```python
import os
import cv2
import numpy as np
from tqdm import tqdm

class HighlightFaceDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def generate_hfd(self, images_path, masks_path, save_path):
        images = glob.glob(os.path.join(images_path, '*.jpg'))
        masks = glob.glob(os.path.join(masks_path, '*.png'))

        os.makedirs(save_path, exist_ok=True)

        for img_path, mask_path in tqdm(zip(images, masks), total=len(images)):
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Ensure the mask is binary
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Save image and mask to HFD dataset directory
            base_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(save_path, 'images', base_name), image)
            cv2.imwrite(os.path.join(save_path, 'masks', base_name.replace('.jpg', '.png')), mask)

        print("HFD generation completed.")

# Example usage
hfd = HighlightFaceDataset('./HFD')
hfd.generate_hfd('./original_images', './highlight_masks', './HFD')
```
**Explanation**:

- The HighlightFaceDataset class provides a method to generate the HFD dataset by combining images with their corresponding highlight masks.
- The generate_hfd method loads the images and masks, ensures the masks are binary, and saves them in the specified HFD directory.
##### Using U-Net for Highlight Transfer
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class HighlightDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(images_path, '*.jpg')))
        self.masks = sorted(glob.glob(os.path.join(masks_path, '*.png')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

class UNet(nn.Module):
    # Define the U-Net architecture here
    def __init__(self):
        super(UNet, self).__init__()
        # Define layers

    def forward(self, x):
        # Define forward pass
        return x

def train_unet(dataset, model, epochs=20, batch_size=8, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in dataloader:
            images = images.float().cuda()
            masks = masks.float().cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}")

# Example usage
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = HighlightDataset('./HFD/images', './HFD/masks', transform=transform)
model = UNet().cuda()
train_unet(dataset, model)
print("Training completed.")
```
**Explanation**:

- The HighlightDataset class provides a PyTorch dataset for loading images and masks from the HFD dataset, applying necessary transformations.
- The UNet class defines the U-Net architecture for highlight transfer. The specific layers and forward pass need to be defined as per U-Net's architecture.
- The train_unet function trains the U-Net model on the highlight dataset using binary cross-entropy loss and the Adam optimizer.
##### Performing Highlight Transfer
```python
class HighlightTransfer:
    def __init__(self, unet_checkpoint):
        self.unet = UNet().cuda()
        self.unet.load_state_dict(torch.load(unet_checkpoint))
        self.unet.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def transfer_highlight(self, source_image_path, reference_image_path):
        source_image = cv2.imread(source_image_path)
        reference_image = cv2.imread(reference_image_path)
        source_tensor = self.transform(source_image).unsqueeze(0).cuda()
        reference_tensor = self.transform(reference_image).unsqueeze(0).cuda()

        with torch.no_grad():
            highlight_mask = self.unet(reference_tensor)

        highlight_mask = torch.sigmoid(highlight_mask).cpu().numpy()[0, 0]
        highlight_mask = (highlight_mask > 0.5).astype(np.uint8) * 255

        result = cv2.bitwise_and(reference_image, reference_image, mask=highlight_mask)
        result = cv2.addWeighted(source_image, 1, result, 0.5, 0)

        return result

# Example usage
highlight_transfer = HighlightTransfer('path_to_unet_checkpoint')
result_image = highlight_transfer.transfer_highlight('path_to_source_image.jpg', 'path_to_reference_image.jpg')
cv2.imwrite('highlight_transferred_image.jpg', result_image)
print("Highlight transfer completed and saved as 'highlight_transferred_image.jpg'.")
```
**Explanation**:

- The HighlightTransfer class initializes the U-Net model for highlight transfer using a pre-trained checkpoint.
- The transfer_highlight method processes the source and reference images, generates the highlight mask using U-Net, and combines the source image with the highlighted areas from the reference image.

This structure provides a detailed approach to highlight transfer, including generating and loading the HFD dataset, training a U-Net model, and performing highlight transfer using the trained U-Net model. This ensures accurate and aesthetically pleasing transfer of highlight effects.
### 3. Pattern Transfer
#### Overview

- **Uses Resnet50MultiScale, ViT, and W-DA models to extract and transfer complex patterns**: This section describes how to use advanced models like Resnet50MultiScale, Vision Transformer (ViT), and Wasserstein Domain Adaptation (W-DA) to effectively transfer complex makeup patterns from a reference image to a source image.
#### Code Analysis
##### Loading and Initializing the Pattern Transfer Network
```python
import segmentation_models_pytorch as smp
import torch
from torchvision import transforms

class PatternTransfer:
    def __init__(self, args):
        # Initialize the segmentation model based on user arguments
        if args.decoder == "fpn":
            self.model = smp.FPN(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        elif args.decoder == "unet":
            self.model = smp.Unet(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        elif args.decoder == "deeplabv3":
            self.model = smp.DeepLabV3(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        else:
            self.model = smp.PSPNet(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
        self.model = self.model.to(args.device)
        self.model.eval()

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0).to(args.device)
        return image

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

# Example usage
args = get_args()
pattern_transfer = PatternTransfer(args)
pattern_transfer.load_checkpoint('path_to_pattern_checkpoint')
image_tensor = pattern_transfer.preprocess_image('path_to_image.jpg')
print(f"Preprocessed Image Tensor Shape: {image_tensor.shape}")
```
**Explanation**:

- The PatternTransfer class initializes the segmentation model based on the provided arguments, which specify the encoder, decoder, and other parameters.
- The preprocess_image method preprocesses an input image for the model.
- The load_checkpoint method loads a pre-trained model checkpoint.
##### Using Resnet50MultiScale and ViT for Pattern Transfer
```python
from torchvision.models import resnet50
from transformers import ViTFeatureExtractor, ViTModel

class PatternTransfer:
    def __init__(self, args):
        # Initialize the segmentation model based on user arguments
        if args.decoder == "fpn":
            self.model = smp.FPN(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        elif args.decoder == "unet":
            self.model = smp.Unet(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        elif args.decoder == "deeplabv3":
            self.model = smp.DeepLabV3(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )
        else:
            self.model = smp.PSPNet(
                encoder_name=args.encoder,
                encoder_weights=args.encoder_weights,
                classes=len(args.classes),
                activation=args.activation,
            )

        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)
        self.model = self.model.to(args.device)
        self.model.eval()

        # Initialize Resnet50 and Vision Transformer
        self.resnet = resnet50(pretrained=True).to(args.device)
        self.vit_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').to(args.device)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image = transform(image).unsqueeze(0).to(args.device)
        return image

    def extract_patterns(self, image_tensor):
        # Extract patterns using Resnet50
        resnet_features = self.resnet(image_tensor)

        # Extract patterns using Vision Transformer (ViT)
        vit_inputs = self.vit_extractor(images=image_tensor.squeeze(0).cpu(), return_tensors="pt").to(args.device)
        vit_outputs = self.vit_model(**vit_inputs)

        return resnet_features, vit_outputs.last_hidden_state

    def transfer_patterns(self, source_image_path, reference_image_path):
        source_image = self.preprocess_image(source_image_path)
        reference_image = self.preprocess_image(reference_image_path)

        # Extract patterns from both source and reference images
        source_resnet_features, source_vit_features = self.extract_patterns(source_image)
        reference_resnet_features, reference_vit_features = self.extract_patterns(reference_image)

        # Perform pattern transfer logic here (combining features, etc.)

        # Placeholder for the final transferred image
        transferred_image = source_image

        return transferred_image

# Example usage
args = get_args()
pattern_transfer = PatternTransfer(args)
pattern_transfer.load_checkpoint('path_to_pattern_checkpoint')
transferred_image = pattern_transfer.transfer_patterns('path_to_source_image.jpg', 'path_to_reference_image.jpg')
cv2.imwrite('pattern_transferred_image.jpg', transferred_image.cpu().squeeze().permute(1, 2, 0).numpy())
print("Pattern transfer completed and saved as 'pattern_transferred_image.jpg'.")
```
**Explanation**:

- The PatternTransfer class is extended to include the initialization of Resnet50 and Vision Transformer (ViT).
- The extract_patterns method extracts features from the input image using both Resnet50 and ViT.
- The transfer_patterns method preprocesses the source and reference images, extracts patterns using Resnet50 and ViT, and performs the pattern transfer logic.

This structure provides a detailed approach to pattern transfer, including loading and initializing the pattern transfer network, and using Resnet50MultiScale and ViT for extracting and transferring complex patterns. The code combines these advanced models to achieve effective and aesthetically pleasing makeup pattern transfers.
### Experiments
![Comparison of Transfer Results](/blogs1/fig11.jpg "Comparison of Transfer Results")
### 1. Quantitative Experiment
#### Overview:

- **Performs quantitative evaluation using metrics like SSIM, LPIPS, and FID**: The quantitative evaluation involves using metrics such as Structural Similarity Index (SSIM), Learned Perceptual Image Patch Similarity (LPIPS), and Frechet Inception Distance (FID) to objectively measure the quality of the transferred makeup.
- **Compares the transfer effects of different models**: By comparing these metrics across different models, we can determine which model performs best in terms of maintaining perceptual similarity, structural integrity, and overall quality.
#### Code Analysis
##### Defining and Calculating Evaluation Metrics
```python
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as msssim
import lpips
from torchvision import transforms
from PIL import Image

# Initialize LPIPS metric
lpips_fn = lpips.LPIPS(net='alex').cuda()

def calculate_ssim(img1, img2):
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    return ssim(img1, img2, multichannel=True, data_range=img1.max() - img1.min())

def calculate_lpips(img1, img2):
    return lpips_fn(img1, img2).item()

def calculate_fid(real_activations, generated_activations):
    mu1, sigma1 = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = np.sqrt(sigma1.dot(sigma2))
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def extract_activations(model, images):
    activations = []
    for img in images:
        img = img.unsqueeze(0).cuda()
        with torch.no_grad():
            activations.append(model(img).cpu().numpy().flatten())
    return np.array(activations)

# Example usage of evaluation metrics
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

img1 = transform(Image.open('path_to_image1.jpg')).cuda()
img2 = transform(Image.open('path_to_image2.jpg')).cuda()

ssim_score = calculate_ssim(img1, img2)
lpips_score = calculate_lpips(img1, img2)

print(f"SSIM: {ssim_score}")
print(f"LPIPS: {lpips_score}")
```
**Explanation**:

- The code defines functions for calculating SSIM, LPIPS, and FID metrics.
- The calculate_ssim function computes the Structural Similarity Index between two images.
- The calculate_lpips function computes the Learned Perceptual Image Patch Similarity using the LPIPS metric.
- The calculate_fid function computes the Frechet Inception Distance given the activations of a model.
- The extract_activations function extracts activations from a model for a set of images.
- An example usage of these metrics is provided.
##### Evaluating and Comparing Model Performance
```python
class ModelEvaluator:
    def __init__(self, model, reference_images, generated_images):
        self.model = model
        self.reference_images = reference_images
        self.generated_images = generated_images
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.lpips_fn = lpips.LPIPS(net='alex').cuda()

    def evaluate(self):
        ssim_scores = []
        lpips_scores = []
        fid_scores = []

        reference_images = [self.transform(Image.open(img)).cuda() for img in self.reference_images]
        generated_images = [self.transform(Image.open(img)).cuda() for img in self.generated_images]

        for ref_img, gen_img in zip(reference_images, generated_images):
            ssim_scores.append(calculate_ssim(ref_img, gen_img))
            lpips_scores.append(calculate_lpips(ref_img, gen_img))

        real_activations = extract_activations(self.model, reference_images)
        generated_activations = extract_activations(self.model, generated_images)
        fid_score = calculate_fid(real_activations, generated_activations)

        return {
            "SSIM": np.mean(ssim_scores),
            "LPIPS": np.mean(lpips_scores),
            "FID": fid_score
        }

# Example usage
reference_images = ['path_to_reference_image1.jpg', 'path_to_reference_image2.jpg']
generated_images = ['path_to_generated_image1.jpg', 'path_to_generated_image2.jpg']
model = torchvision.models.inception_v3(pretrained=True).cuda()

evaluator = ModelEvaluator(model, reference_images, generated_images)
results = evaluator.evaluate()

print(f"SSIM: {results['SSIM']}")
print(f"LPIPS: {results['LPIPS']}")
print(f"FID: {results['FID']}")
```
**Explanation**:

- The ModelEvaluator class takes a model and lists of reference and generated images for evaluation.
- The evaluate method calculates SSIM, LPIPS, and FID scores for the images.
- The method preprocesses the images, computes the scores, and returns the average SSIM and LPIPS scores along with the FID score.
- An example usage demonstrates how to instantiate the ModelEvaluator class and evaluate model performance on a set of images.

This structure provides a comprehensive framework for performing quantitative experiments using SSIM, LPIPS, and FID metrics to evaluate and compare the performance of different models in makeup transfer tasks.
### 2. Qualitative Experiment
![Control Group](/blogs1/fig13.jpg "Control Group")![Comparison of Different Models](/blogs1/fig12.jpg "Comparison of Different Models")

- **Overview**:
   - Conducts qualitative evaluation through visual effects and user feedback.
   - Compares makeup transfer results of different models.
### Ablation Study
![Ablation Study Result](/blogs1/fig14.jpg "Ablation Study Result")
#### 1. Validation of Makeup Adaptation Model

- **Overview**:
   - Validates the contribution of the makeup adaptation model.
   - Compares effects with and without the adaptation model.
- **Code Analysis**:
   - Designing and conducting A/B experiments.
   - Collecting and analyzing experimental data.
#### 2. Validation of Color Transfer Branch

- **Overview**:
   - Validates the contribution of the color transfer branch.
   - Compares effects of different color transfer strategies.
- **Code Analysis**:
   - Removing or replacing the color transfer module for comparison experiments.
   - Analyzing experimental results.
#### 3. Validation of Pattern Transfer Branch

- **Overview**:
   - Validates the contribution of the pattern transfer branch.
   - Compares effects of different pattern transfer strategies.
- **Code Analysis**:
   - Removing or replacing the pattern transfer module for comparison experiments.
   - Analyzing experimental results.
### Conclusion

- **Overview**:
   - Summarizes the innovations and contributions of the FSAMT framework.
   - Proposes future improvements.
- **Code Analysis**:
   - Summarizing and reflecting on experimental results.
   - Suggesting potential improvements and extensions for future work.

![fig2.jpg](/blogs1/fig2.jpg)
