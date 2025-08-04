# Signs_Detector_ResNet50
This project implements ResNet-50, a deep convolutional neural network (CNN) architecture, for image classification. ResNet-50 is a variant of the Residual Network (ResNet) family, known for its use of skip connections (identity blocks) to mitigate the vanishing gradient problem in deep networks.

![Signs_Detector_ResNet50](https://img2020.cnblogs.com/blog/817161/202006/817161-20200616152820993-2136285971.png)


# ResNet-50 Image Classification

## 📌 Overview
This project implements **ResNet-50**, a deep convolutional neural network (CNN) for image classification, using Keras with TensorFlow backend. The model includes identity and convolutional blocks with skip connections to mitigate vanishing gradients in deep networks.

## 🚀 Features
- **ResNet-50 Architecture**:  
  - Identity blocks (`identity_block`) and convolutional blocks (`convolutional_block`).  
  - Skip connections for stable training of deep networks.  
- **Customizable Input/Output**: Supports flexible input shapes and number of classes.  
- **Preprocessing**: Includes image normalization and one-hot label encoding.  
- **Training & Evaluation**: Tracks accuracy/loss during training and testing.  

## 🛠 Installation
### Dependencies
Ensure you have the following installed:
```bash
pip install numpy tensorflow keras matplotlib scipy pydot ipythonDataset
The code assumes a dataset loaded via resnets_utils.py (not provided here). Replace with your dataset or ensure load_dataset() and convert_to_one_hot() are defined.
```
Dataset
The code assumes a dataset loaded via resnets_utils.py (not provided here). Replace with your dataset or ensure load_dataset() and convert_to_one_hot() are defined.

🏷 Usage
1. Model Initialization
```bash
model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
2. Data Preparation
```bash
# Example: Normalize and one-hot encode
X_train = X_train_orig / 255.0
Y_train = convert_to_one_hot(Y_train_orig, 6).T
```
3. Training
```bash
model.fit(X_train, Y_train, epochs=5, batch_size=32)
```
4. Evaluation
```bash
preds = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {preds[1] * 100:.2f}%")
```
5. Model Summary
```bash
model.summary()  # Prints architecture details
```
📂 Code Structure

identity_block(X, f, filters, stage, block):
Implements the identity block with skip connections (no stride change).

convolutional_block(X, f, filters, stage, block, s=2):
Implements the convolutional block with a stride s for downsampling.

ResNet50(input_shape, classes):
Builds the full ResNet-50 model with 5 stages.

![Signs_Detector_ResNet50](images/resnet_kiank.png)  

📊 Example Output

number of training examples = 1080  
number of test examples = 120  
X_train shape: (1080, 64, 64, 3)  
Y_train shape: (1080, 6)  
Test Accuracy: 85.42%  

## 📜 Model Architecture

| Layer (type)                | Output Shape      | Param # | Connected to               |
|-----------------------------|-------------------|---------|----------------------------|
| input_1 (InputLayer)        | (None, 64, 64, 3) | 0       | -                          |
| ...                         | ...               | ...     | ...                        |
| avg_pool (AveragePooling2D) | (None, 1, 1, 2048)| 0       | activation_49[0][0]        |
| fc6 (Dense)                 | (None, 6)         | 12,294  | flatten_1[0][0]            |

Total params: 23,600,006  
Trainable params: 23,546,886  
Non-trainable params: 53,120  

🤝 Contributing

Contributions are welcome! Open an issue or submit a PR for improvements.


