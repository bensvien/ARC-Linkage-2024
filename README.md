**APWSHM 2024 Conference **

Demonstration of Python Script for CNN U-Net model

Strain E_XX CNN pre-trained with 100,000 samples - "model_save2024_v2xx_100k_unet_L_non.keras"
Includes Validation (Original True Set- FE generated in MATLAB data format) - "strain2024_validation_data_100_true.mat"

**Predicting Membrane Strains with a Deep Learning Encoder-Decoder Convolutional Neural Network Architecture Trained on Synthetic Finite Element Data**

Benjamin Steven Vien1,a *, Thomas Kuen2,b , Louis Raymond Francis Rose1,c and Wing Kong Chiu1,d 
1Department of Mechanical and Aerospace Engineering, Monash University, Wellington Rd, Clayton, VIC 3800, Australia 
2Melbourne Water Corporation, 990 La Trobe Street, Docklands, VIC 3008, Australia
aben.vien@monash.edu, b thomas.kuen@melbournewater.com.au, cfrancis.rose@monash.edu, dwing.king.chiu@monash.edu
Keywords: Deep learning, membrane, strain, convolution neural network, synthetic data, finite element.
ABSTRACT
Melbourne Water Corporation's Western Treatment Plant, located in Werribee, Victoria, Australia, is pioneering the use of advanced technologies, including artificial intelligence, to enhance asset management practices. This is particularly important for the structural health monitoring (SHM) of anaerobic lagoon floating covers, which are crucial for biogas collection but can be compromised by scum accumulation underneath. This study introduces a novel encoder-decoder network within a convolutional neural network (CNN) framework, designed to predict strain distributions on deformed membranes. Due to the limited availability of real-world data, finite element analysis was utilised to generate synthetic data consisting of displacements and strain fields for model training. The study investigates the optimal quantity of synthetic samples needed for accurate predictions and discusses the proposed CNN architecture and data preparation techniques. The findings indicate that a dataset of at least 10,000 synthetic training samples is required to accurately predict strain distributions, which represents a significant improvement by orders of magnitude compared to using only 100 and 1000 samples. Furthermore, refinement learning methods were demonstrated, where a pretrained CNN model is further trained on a new dataset with lower strain variability. The results indicate that refining the pretrained model with frozen (fixed) weights in the encoder network yields better accuracy in predicting strain, at least 2.3 times better than those without frozen weights. However, the refined model without frozen weights retains more information from the original dataset and is more consistent in predicting strain profiles. The results suggest a high-quality, representative training dataset relating to the application of interest is essential for effective machine learning. These findings lay a fundamental basis for implementing practical deep learning approaches and further utilising unmanned aerial vehicle-based imagery for effective SHM of highly valuable assets.
