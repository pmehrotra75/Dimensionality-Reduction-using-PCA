
# Dimensionality Reduction using PCA

This project demonstrates the use of dimensionality reduction via Principal Component Analysis (PCA) to speed up the training of a Random Forest Classifier on a high-dimensional dataset. The goal is to achieve significant training time improvements while maintaining high predictive accuracy.

## Project Overview

1. **Dataset**:

   - The dataset used is *High Dimensional Datascape*, available on Kaggle:  
     [High Dimensional Datascape Dataset](https://www.kaggle.com/datasets/krishd123/high-dimensional-datascape/data).  
   - The data was downloaded and saved in the `data` folder.

2. **Dimensionality Reduction**:

   - Principal Component Analysis (PCA) was applied to reduce the number of features. This technique minimizes redundancy in the data while retaining most of the variability. 

3. **Classification**:

   - Two Random Forest models were trained:
     - **Model 1**: Using the original dataset.
     - **Model 2**: Using the PCA-transformed dataset with reduced dimensions.

4. **Results**:

   - Model 2 achieved a **70% reduction in training time** compared to Model 1 while retaining up to **90% of Model 1's accuracy**.

## Repository Structure

```
High-Dimensional-Classification/
│
├── data/
│   └── high_dimensional_datascape.csv     # Dataset file
│
├── output/
│   └── result.txt                         # File containing statistics
│
├── src/
│   └── main.py                            # Main Python script for training models
│
├── README.md                              # Project documentation (this file)
└── requirements.txt                       # Python dependencies
```

## Usage Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/high-dimensional-classification.git
   cd high-dimensional-classification
   ```

2. **Install Dependencies**:

   Create a virtual environment and install the required packages:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Windows, use venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Code**:
   Execute the main script from the `src` folder:
   ```bash
   python src/main.py
   ```

4. **View Results**:
   The statistics (accuracy and training time) are saved in `output/result.txt`.

## Results

- **Model 1 (Original Data)**:
  - Training Time: `0.21 seconds`
  - Accuracy: `94.2%`

- **Model 2 (PCA-reduced Data)**:
  - Training Time: `0.15 seconds` (71% faster than Model 1)
  - Accuracy: `89.86%`

## Citation

### Dataset
The dataset used in this project is from Kaggle:  
[High Dimensional Datascape Dataset](https://www.kaggle.com/datasets/krishd123/high-dimensional-datascape/data).

### Learning Resource
This project leveraged concepts learned from the Coursera guided project:  
[Principal Component Analysis with NumPy](https://www.coursera.org/projects/principal-component-analysis-numpy).
