# iDetect-AI-powered-Retinal-Disease-Detection-from-Fundus-Images

## Project Metadata
### Authors
- **Team:** Shahbaaz Ahmed Sadiq, Fahad Alothman
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** SABIC, ARAMCO and KFUPM (write your institution name, and/or KFUPM)

## Introduction
Retinal diseases such as diabetic retinopathy, glaucoma, and age-related macular degeneration are among the leading causes of vision loss worldwide. Detecting these conditions early through color fundus imaging is critical, since timely treatment can often prevent permanent damage. However, manual diagnosis is not only time-consuming but also heavily dependent on the availability and expertise of ophthalmologists, which creates challenges in many parts of the world. Advances in artificial intelligence, particularly in deep learning, have opened new possibilities for automating this process and making retinal disease screening more accurate and accessible. Building on recent research in vision–language models, this project, titled iDetect: AI powered Retinal Disease Detection from Fundus Images, focuses on developing an intelligent system that can classify retinal diseases using the Peacein color fundus eye dataset. By combining state-of-the-art pretrained models with fine-tuning for specific conditions, the project aims to create a tool that could support doctors in making faster and more reliable diagnoses, ultimately helping patients receive timely care.

## Problem Statement
Retinal diseases such as diabetic retinopathy, glaucoma, age-related macular degeneration, and central serous chorioretinopathy cause structural changes in the fundus that may be subtle and difficult to detect, especially in early stages. Manual screening by expert ophthalmologists is laborious, time-consuming, and subject to intra-observer variability. Moreover, many regions lack access to specialists, increasing risk of late diagnosis. The problem, then, is to develop an automated method that can reliably classify or detect multiple retinal pathologies from color fundus photographs (CFPs) with high sensitivity and specificity across diverse patient populations and imaging conditions.

From a machine learning angle, the challenge is to build a model that generalizes well across multiple diseases, handles class imbalance, copes with differences in imaging devices and patient populations, and ideally captures good feature representations so that downstream tasks (classification, multi-label detection, disease localization) perform robustly. In particular, one would like to explore how to leverage large-scale pretraining, multi-modal learning (e.g. combining images and diagnostic reports), and transfer learning to overcome limited annotated data in each disease category.

## Application Area and Project Domain
Our project lies at the intersection of medical image analysis (ophthalmic imaging) and AI/vision + clinical decision support systems. The application area is ophthalmology / retinal disease screening.
The domain is diagnostic support: We aim to assist or augment clinicians in screening retinal fundus images, flagging high-risk cases or referring those needing further examination.

## What is the paper trying to do, and what are you planning to do?
### What the RET-CLIP Paper Does

The RET-CLIP paper presents a foundation-style vision-language model tailored for retinal imaging. It uses a large-scale dataset of color fundus photographs paired with clinical diagnostic reports (text) to pretrain a CLIP-style embedding: mapping images and corresponding textual diagnostic descriptions into a shared embedding space. They adopt a tripartite optimization scheme at the left-eye, right-eye, and patient level to reflect clinical relationships. After pretraining, RET-CLIP is fine-tuned or adapted for downstream tasks across multiple retinal disease classification benchmarks (diabetic retinopathy, glaucoma, multi-label disease diagnosis) and achieves state-of-the-art performance across eight datasets. The strength lies in learning strong, generalizable retinal image features via vision-text co-training, so that fewer labels are needed for downstream tasks.

### What You Intend to Do

This project adapts the RET-CLIP idea using the Peacein/color-fundus-eye dataset (~16k images, 10 classes). Since the original RET-CLIP used a private clinical dataset, we will create synthetic text prompts from class labels (e.g., “fundus image showing diabetic retinopathy”) to form image–text pairs. With these, we will train a mini CLIP-style model via contrastive learning, then fine-tune the image encoder with a classifier head for multi-class retinal disease detection. Finally, we will compare this with a standard image-only baseline (ResNet/ViT) to assess the benefits of vision–language pretraining.

# THE FOLLOWING IS SUPPOSED TO BE DONE LATER

### Project Documents
- **Presentation:** [Project Presentation](/presentation.pptx)
- **Report:** [Project Report](/report.pdf)

### Reference Paper
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

### Reference Dataset
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)


## Project Technicalities

### Terminologies
- **Diffusion Model:** A generative model that progressively transforms random noise into coherent data.
- **Latent Space:** A compressed, abstract representation of data where complex features are captured.
- **UNet Architecture:** A neural network with an encoder-decoder structure featuring skip connections for better feature preservation.
- **Text Encoder:** A model that converts text into numerical embeddings for downstream tasks.
- **Perceptual Loss:** A loss function that measures high-level differences between images, emphasizing perceptual similarity.
- **Tokenization:** The process of breaking down text into smaller units (tokens) for processing.
- **Noise Vector:** A randomly generated vector used to initialize the diffusion process in generative models.
- **Decoder:** A network component that transforms latent representations back into image space.
- **Iterative Refinement:** The process of gradually improving the quality of generated data through multiple steps.
- **Conditional Generation:** The process where outputs are generated based on auxiliary inputs, such as textual descriptions.

### Problem Statements
- **Problem 1:** Achieving high-resolution and detailed images using conventional diffusion models remains challenging.
- **Problem 2:** Existing models suffer from slow inference times during the image generation process.
- **Problem 3:** There is limited capability in performing style transfer and generating diverse artistic variations.

### Loopholes or Research Areas
- **Evaluation Metrics:** Lack of robust metrics to effectively assess the quality of generated images.
- **Output Consistency:** Inconsistencies in output quality when scaling the model to higher resolutions.
- **Computational Resources:** Training requires significant GPU compute resources, which may not be readily accessible.

### Problem vs. Ideation: Proposed 3 Ideas to Solve the Problems
1. **Optimized Architecture:** Redesign the model architecture to improve efficiency and balance image quality with faster inference.
2. **Advanced Loss Functions:** Integrate novel loss functions (e.g., perceptual loss) to better capture artistic nuances and structural details.
3. **Enhanced Data Augmentation:** Implement sophisticated data augmentation strategies to improve the model’s robustness and reduce overfitting.

### Proposed Solution: Code-Based Implementation
This repository provides an implementation of the enhanced stable diffusion model using PyTorch. The solution includes:

- **Modified UNet Architecture:** Incorporates residual connections and efficient convolutional blocks.
- **Novel Loss Functions:** Combines Mean Squared Error (MSE) with perceptual loss to enhance feature learning.
- **Optimized Training Loop:** Reduces computational overhead while maintaining performance.

### Key Components
- **`model.py`**: Contains the modified UNet architecture and other model components.
- **`train.py`**: Script to handle the training process with configurable parameters.
- **`utils.py`**: Utility functions for data processing, augmentation, and metric evaluations.
- **`inference.py`**: Script for generating images using the trained model.

## Model Workflow
The workflow of the Enhanced Stable Diffusion model is designed to translate textual descriptions into high-quality artistic images through a multi-step diffusion process:

1. **Input:**
   - **Text Prompt:** The model takes a text prompt (e.g., "A surreal landscape with mountains and rivers") as the primary input.
   - **Tokenization:** The text prompt is tokenized and processed through a text encoder (such as a CLIP model) to obtain meaningful embeddings.
   - **Latent Noise:** A random latent noise vector is generated to initialize the diffusion process, which is then conditioned on the text embeddings.

2. **Diffusion Process:**
   - **Iterative Refinement:** The conditioned latent vector is fed into a modified UNet architecture. The model iteratively refines this vector by reversing a diffusion process, gradually reducing noise while preserving the text-conditioned features.
   - **Intermediate States:** At each step, intermediate latent representations are produced that increasingly capture the structure and details dictated by the text prompt.

3. **Output:**
   - **Decoding:** The final refined latent representation is passed through a decoder (often part of a Variational Autoencoder setup) to generate the final image.
   - **Generated Image:** The output is a synthesized image that visually represents the input text prompt, complete with artistic style and detail.

## How to Run the Code

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/enhanced-stable-diffusion.git
    cd enhanced-stable-diffusion
    ```

2. **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Train the Model:**
    Configure the training parameters in the provided configuration file and run:
    ```bash
    python train.py --config configs/train_config.yaml
    ```

4. **Generate Images:**
    Once training is complete, use the inference script to generate images.
    ```bash
    python inference.py --checkpoint path/to/checkpoint.pt --input "A surreal landscape with mountains and rivers"
    ```

## Acknowledgments
- **Open-Source Communities:** Thanks to the contributors of PyTorch, Hugging Face, and other libraries for their amazing work.
- **Individuals:** Special thanks to bla, bla, bla for the amazing team effort, invaluable guidance and support throughout this project.
- **Resource Providers:** Gratitude to ABC-organization for providing the computational resources necessary for this project.
