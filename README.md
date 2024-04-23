# CIFAR-10 Image Classification with PyTorch

## Project Overview
This detailed project implements a convolutional neural network (CNN) to classify images from the CIFAR-10 (Canadian Institute for Advanced Research, 10 classes) dataset. Using PyTorch, the model evolves through structured stages, each aimed at enhancing performance through advanced machine learning techniques.

### Dataset
- **Training Dataset**: 50,000 images
- **Testing Dataset**: 10,000 images

### Class Labels
- (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

### Neural Network Architecture
The CNN architecture is developed through several stages:
1. **Initial Setup and Baseline Model**:
   - **Description**: Basic CNN architecture employing stochastic gradient descent (SGD) with a learning rate of 0.01, trained over 10 epochs with 1 intermediate block.
   - **Testing Accuracy Achieved**: 28.46%
2. **Intermediate Stage (Trial 1)**:
   - **Enhancements**: Introduction of advanced data augmentation techniques, increase in batch size from 32 to 64, extension of training to 20 epochs, and expansion to 6 intermediate blocks.
   - **Testing Accuracy Achieved**: 23.57%
3. **Advanced Stage (Final Trial)**:
   - **Optimizations**: Refinement of data augmentation methods, implementation of batch normalization, and dropout regularization. Training involves an AdamW optimizer with a CosineAnnealingLR scheduler across 200 epochs.
   - **Testing Accuracy Achieved**: 86.12%

### Training and Testing
- **Loss Function**: Cross-entropy loss, suitable for multi-class classification tasks.
- **Optimization Techniques**: Transition from SGD to AdamW to enhance optimization efficiency, supported by dynamic learning rate adjustments through StepLR and CosineAnnealingLR schedulers.
- **Regularization Methods**: Dropout and batch normalization to stabilize and improve network training.

### Results
The project effectively demonstrates the step-by-step improvement in model performance:
- **Initial Stage**: Set a robust baseline for understanding basic CNN capabilities.
- **Intermediate Stage**: Increased complexity and training depth surprisingly lead to a decrease in testing accuracy but that could be due to the fact that advanced data augmentation technqiues made it hard for the machine learning model to generalise the unseen images.
- **Advanced Stage**: Achieved optimal performance with comprehensive tuning and regularization, showcasing significant advancements in model accuracy and generalization on unseen data.

## Usage
Follow these steps to replicate or explore the project:
1. Clone the repository.
2. Install the necessary dependencies: `pip install -r requirements.txt`.
3. Execute the Jupyter notebook to train the model and assess performance metrics visually through provided plots.

## Acknowledgments
- Thanks to CIFAR for providing the dataset.
- Gratitude to all educators and peers who supported this project with their insights and feedback.

