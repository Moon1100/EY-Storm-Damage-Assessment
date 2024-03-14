# YoloNAS Model for Cyclone Impact Area Detection

## Overview
This repository contains the code and resources for training a YoloNAS (You Only Look Once Neural Architecture Search) model to detect specific objects within satellite images of cyclone-affected areas. The model is designed to identify undamaged and damaged residential and commercial buildings in the aftermath of a cyclone event.

## Project Structure
- **`checkpoint/`**: Directory to save model checkpoints during training.
- **`dataset/`**: Contains the dataset for training, validation, and testing.
  - **`train/`**: Training images and labels.
  - **`val/`**: Validation images and labels.
  - **`test/`**: Test images and labels.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/YoloNAS-Cyclone-Impact-Detection.git
   cd YoloNAS-Cyclone-Impact-Detection
   ```

## Usage
1. Prepare the dataset by organizing images and labels in the `dataset/` directory.
2. Update the configuration parameters in `config.py` for dataset paths, model settings, and training parameters.
3. Run the training script to train the YoloNAS model:
   ```bash
   python train.py
   ```
4. Evaluate the model performance on the test set:
   ```bash
   python evaluate.py
   ```

## Acknowledgements
- This project is based on the YoloNAS architecture and utilizes the SuperGradients library for training.
- The dataset used for training and evaluation is sourced from satellite images of cyclone-affected areas.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For more details and in-depth documentation, please refer to the code files and comments within the repository. Feel free to reach out for any questions or collaborations. Thank you for your interest in the YoloNAS Model for Cyclone Impact Area Detection!
