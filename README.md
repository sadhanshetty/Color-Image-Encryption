# Color Image Encryption Using Logistic Map

## Overview

This project demonstrates a color image encryption and decryption process using a chaotic logistic map. The encryption algorithm is based on the logistic map, which is a mathematical chaotic system. This program also includes functionality to visualize histograms and autocorrelations of images before and after encryption.

## Features

- **Image Encryption**: Encrypts a color image using a logistic map-based chaotic encryption algorithm.
- **Image Decryption**: Decrypts the encrypted image using the same key.
- **Histogram Plotting**: Visualizes histograms of the original and encrypted images.
- **Autocorrelation Plotting**: Visualizes autocorrelations of the original and encrypted images to demonstrate redundancy reduction.

## Dependencies

The following libraries are required for the project:

- `tkinter` (for the graphical user interface)
- `Pillow` (for image handling)
- `numpy` (for image data manipulation)
- `matplotlib` (for plotting histograms and autocorrelations)

You can install these dependencies using the following command:

```bash
pip install tkinter pillow numpy matplotlib
```

## Getting Started

Follow these steps to run the project on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/color-image-encryption-using-logistic-map.git
cd color-image-encryption-using-logistic-map
```

### 2. Install Dependencies
Run the following command to install the necessary libraries:

```bash
pip install -r requirements.txt
```

If no requirements.txt file is available, install the libraries manually:

```bash
pip install pillow numpy matplotlib
```

### 3. Run the Application
To launch the graphical user interface for encrypting and decrypting images, run the following command:

```bash
python main.py
```
Replace ```main.py``` with the actual name of your Python file if it's different.


## How to Use

### GUI Interface

- **Select an Image**: Click the **Select Image** button to browse and select the image you want to encrypt or decrypt.
- **Enter Encryption Key**: Input a key (string) in the key field.
- **Encrypt the Image**: Click the **Encrypt** button to encrypt the selected image. The encrypted image will be displayed and saved with the suffix `_LogisticEnc.png`.
- **Decrypt the Image**: Click the **Decrypt** button to decrypt an encrypted image. The decrypted image will be saved with the suffix `_LogisticDec.png`.
- **View Histograms**: Click the **Plot Histogram** button to view the color histogram of both the original and encrypted images.
- **View Autocorrelations**: Click the **Plot Autocorrelation** button to visualize the autocorrelation of both the original and encrypted images.

### Command-Line Usage (Optional)

You can modify the code to work via command-line inputs if needed, but the current version is focused on GUI interaction.


## Example

### Original Image:
![lena](https://github.com/user-attachments/assets/cf1dd1a9-6eac-4663-8064-f823ef7eb10f)


### Encrypted Image:
![lena_LogisticEnc](https://github.com/user-attachments/assets/51d953e8-b78c-4749-9c19-4dc342f35263)


## Contributing

If you wish to contribute to the project, feel free to fork the repository and create a pull request. Contributions in the form of bug fixes, feature additions, or code optimizations are welcome.

## License

This project is licensed under the MIT License.

## Acknowledgements

- The encryption method is inspired by the use of chaotic maps for image encryption.
- Special thanks to open-source contributors for providing libraries like **Pillow**, **NumPy**, and **Matplotlib** that make this project possible.

## Screenshots

### GUI Window:
![Screenshot 1](https://github.com/user-attachments/assets/8ef30cd4-29a6-401f-8dff-27d5c504963e)

### Original Image:
![lena](https://github.com/user-attachments/assets/bb454754-e2ad-44bf-8f2f-5f0b3d2c8d76)

### Encrypted Image:
![lena_LogisticEnc](https://github.com/user-attachments/assets/f633743d-e51a-4914-a59c-35cbd7539bc5)

### Histogram of Original Image:
![Figure_1](https://github.com/user-attachments/assets/9677dc20-a943-4bbd-8e90-8e91e06320ba)

### Histogram Of Encrypted Image:
![Figure_2](https://github.com/user-attachments/assets/a6da655b-7fd9-4f34-a4e8-842c83479ba3)
