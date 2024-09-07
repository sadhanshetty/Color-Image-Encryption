import os
from tkinter import Tk, Label, Button, Entry, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

def getImageMatrix(imageName):
    im = Image.open(imageName)
    pix = im.load()
    color = 1
    if type(pix[0, 0]) == int:
        color = 0
    image_size = im.size
    image_matrix = []
    for width in range(int(image_size[0])):
        row = []
        for height in range(int(image_size[1])):
            row.append(pix[width, height])
        image_matrix.append(row)
    return image_matrix, image_size[0], image_size[1], color

def prepare_key(key):
    key_list = [ord(x) for x in key]
    while len(key_list) < 13:
        key_list.append(0)  # Pad with zeros if the key is too short
    return key_list

def logistic_encryption(imageName, key):
    key_list = prepare_key(key)
    N = 256
    G = [key_list[0:4], key_list[4:8], key_list[8:12]]
    g = []
    R = 1
    for i in range(1, 4):
        s = 0
        for j in range(1, 5):
            s += G[i - 1][j - 1] * (10 ** (-j))
        g.append(s)
        R = (R * s) % 1

    L = (R + key_list[12] / 256) % 1
    S_x = round(((g[0] + g[1] + g[2]) * (10 ** 4) + L * (10 ** 4)) % 256)
    V1 = sum(key_list)
    V2 = key_list[0]
    for i in range(1, len(key_list)):
        V2 = V2 ^ key_list[i]
    V = V2 / V1

    L_y = (V + key_list[12] / 256) % 1
    S_y = round((V + V2 + L_y * 10 ** 4) % 256)
    C1_0 = S_x
    C2_0 = S_y
    C = round((L * L_y * 10 ** 4) % 256)
    C_r = round((L * L_y * 10 ** 4) % 256)
    C_g = round((L * L_y * 10 ** 4) % 256)
    C_b = round((L * L_y * 10 ** 4) % 256)
    x = 4 * (S_x) * (1 - S_x)
    y = 4 * (S_y) * (1 - S_y)

    imageMatrix, dimensionX, dimensionY, color = getImageMatrix(imageName)
    LogisticEncryptionIm = []
    for i in range(dimensionX):
        row = []
        for j in range(dimensionY):
            while x < 0.8 and x > 0.2:
                x = 4 * x * (1 - x)
            while y < 0.8 and y > 0.2:
                y = 4 * y * (1 - y)
            x_round = round((x * (10 ** 4)) % 256)
            y_round = round((y * (10 ** 4)) % 256)
            C1 = x_round ^ ((key_list[0] + x_round) % N) ^ ((C1_0 + key_list[1]) % N)
            C2 = y_round ^ ((key_list[2] + y_round) % N) ^ ((C2_0 + key_list[3]) % N)
            if color:
                C_r = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                            (key_list[6] + imageMatrix[i][j][0]) % N) ^ ((C_r + key_list[7]) % N)
                C_g = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                            (key_list[6] + imageMatrix[i][j][1]) % N) ^ ((C_g + key_list[7]) % N)
                C_b = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                            (key_list[6] + imageMatrix[i][j][2]) % N) ^ ((C_b + key_list[7]) % N)
                row.append((C_r, C_g, C_b))
                C = C_r
            else:
                C = ((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ (
                            (key_list[6] + imageMatrix[i][j]) % N) ^ ((C + key_list[7]) % N)
                row.append(C)

            x = (x + C / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
            y = (y + C / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
            for ki in range(len(key_list) - 1):
                key_list[ki] = (key_list[ki] + key_list[-1]) % 256
                key_list[-1] = key_list[-1] ^ key_list[ki]
        LogisticEncryptionIm.append(row)

    im = Image.new("L", (dimensionX, dimensionY))
    if color:
        im = Image.new("RGB", (dimensionX, dimensionY))
    else:
        im = Image.new("L", (dimensionX, dimensionY))  # L is for Black and white pixels

    pix = im.load()
    for x in range(dimensionX):
        for y in range(dimensionY):
            pix[x, y] = LogisticEncryptionIm[x][y]
    output_path = imageName.split('.')[0] + "_LogisticEnc.png"
    im.save(output_path, "PNG")
    return output_path

def logistic_decryption(imageName, key):
    key_list = prepare_key(key)
    N = 256
    G = [key_list[0:4], key_list[4:8], key_list[8:12]]
    g = []
    R = 1
    for i in range(1, 4):
        s = 0
        for j in range(1, 5):
            s += G[i - 1][j - 1] * (10 ** (-j))
        g.append(s)
        R = (R * s) % 1

    L_x = (R + key_list[12] / 256) % 1
    S_x = round(((g[0] + g[1] + g[2]) * (10 ** 4) + L_x * (10 ** 4)) % 256)
    V1 = sum(key_list)
    V2 = key_list[0]
    for i in range(1, len(key_list)):
        V2 = V2 ^ key_list[i]
    V = V2 / V1

    L_y = (V + key_list[12] / 256) % 1
    S_y = round((V + V2 + L_y * 10 ** 4) % 256)
    C1_0 = S_x
    C2_0 = S_y

    C = round((L_x * L_y * 10 ** 4) % 256)
    I_prev = C
    I_prev_r = C
    I_prev_g = C
    I_prev_b = C
    I = C
    I_r = C
    I_g = C
    I_b = C
    x_prev = 4 * (S_x) * (1 - S_x)
    y_prev = 4 * (L_x) * (1 - S_y)
    x = x_prev
    y = y_prev
    imageMatrix, dimensionX, dimensionY, color = getImageMatrix(imageName)

    logisticDecryptedImage = []
    for i in range(dimensionX):
        row = []
        for j in range(dimensionY):
            while x < 0.8 and x > 0.2:
                x = 4 * x * (1 - x)
            while y < 0.8 and y > 0.2:
                y = 4 * y * (1 - y)
            x_round = round((x * (10 ** 4)) % 256)
            y_round = round((y * (10 ** 4)) % 256)
            C1 = x_round ^ ((key_list[0] + x_round) % N) ^ ((C1_0 + key_list[1]) % N)
            C2 = y_round ^ ((key_list[2] + y_round) % N) ^ ((C2_0 + key_list[3]) % N)
            if color:
                I_r = ((((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ ((I_prev_r + key_list[7]) % N) ^
                        imageMatrix[i][j][0]) + N - key_list[6]) % N
                I_g = ((((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ ((I_prev_g + key_list[7]) % N) ^
                        imageMatrix[i][j][1]) + N - key_list[6]) % N
                I_b = ((((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ ((I_prev_b + key_list[7]) % N) ^
                        imageMatrix[i][j][2]) + N - key_list[6]) % N
                I_prev_r = imageMatrix[i][j][0]
                I_prev_g = imageMatrix[i][j][1]
                I_prev_b = imageMatrix[i][j][2]
                row.append((I_r, I_g, I_b))
                x = (x + imageMatrix[i][j][0] / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
                y = (y + imageMatrix[i][j][0] / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
            else:
                I = ((((key_list[4] + C1) % N) ^ ((key_list[5] + C2) % N) ^ ((I_prev + key_list[7]) % N) ^
                      imageMatrix[i][j]) + N - key_list[6]) % N
                I_prev = imageMatrix[i][j]
                row.append(I)
                x = (x + imageMatrix[i][j] / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
                y = (y + imageMatrix[i][j] / 256 + key_list[8] / 256 + key_list[9] / 256) % 1
            for ki in range(len(key_list) - 1):
                key_list[ki] = (key_list[ki] + key_list[-1]) % 256
                key_list[-1] = key_list[-1] ^ key_list[ki]
        logisticDecryptedImage.append(row)

    if color:
        im = Image.new("RGB", (dimensionX, dimensionY))
    else:
        im = Image.new("L", (dimensionX, dimensionY))  # L is for Black and white pixels
    pix = im.load()
    for x in range(dimensionX):
        for y in range(dimensionY):
            pix[x, y] = logisticDecryptedImage[x][y]
    output_path = imageName.split('_')[0] + "_LogisticDec.png"
    im.save(output_path, "PNG")
    return output_path

def plot_histogram(imageName):
    im = Image.open(imageName)
    im_array = np.array(im)
    if len(im_array.shape) == 3:
        plt.figure()
        plt.title("Histogram for Color Image")
        plt.hist(im_array[:,:,0].ravel(), bins=256, color='red', alpha=0.5, label='Red Channel')
        plt.hist(im_array[:,:,1].ravel(), bins=256, color='green', alpha=0.5, label='Green Channel')
        plt.hist(im_array[:,:,2].ravel(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    else:
        plt.figure()
        plt.title("Histogram for Grayscale Image")
        plt.hist(im_array.ravel(), bins=256, color='black', alpha=0.5)
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
        plt.show()

def plot_autocorrelation(imageName):
    im = Image.open(imageName)
    im_array = np.array(im)
    if len(im_array.shape) == 3:
        im_array = np.mean(im_array, axis=2)
    autocorr = np.correlate(im_array.ravel(), im_array.ravel(), mode='full')
    autocorr /= np.max(autocorr)
    plt.figure()
    plt.title("Autocorrelation of Image")
    plt.plot(autocorr)
    plt.xlabel('Offset')
    plt.ylabel('Autocorrelation')
    plt.show()

def show_image(imageName):
    im = Image.open(imageName)
    im.show()

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".png;.jpg;.jpeg;.bmp")])
    if file_path:
        entry_image_path.delete(0, "end")
        entry_image_path.insert(0, file_path)

def encrypt_image():
    image_path = entry_image_path.get()
    key = entry_key.get()
    if not image_path or not key:
        messagebox.showerror("Error", "Please provide both image path and key.")
        return
    try:
        encrypted_image_path = logistic_encryption(image_path, key)
        messagebox.showinfo("Success", f"Image encrypted and saved to {encrypted_image_path}")
        show_image(encrypted_image_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def decrypt_image():
    image_path = entry_image_path.get()
    key = entry_key.get()
    if not image_path or not key:
        messagebox.showerror("Error", "Please provide both image path and key.")
        return
    try:
        decrypted_image_path = logistic_decryption(image_path, key)
        messagebox.showinfo("Success", f"Image decrypted and saved to {decrypted_image_path}")
        show_image(decrypted_image_path)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def plot_histograms():
    image_path = entry_image_path.get()
    if not image_path:
        messagebox.showerror("Error", "Please provide an image path.")
        return
    plot_histogram(image_path)
    plot_histogram(image_path.split('.')[0] + "_LogisticEnc.png")

def plot_autocorrelations():
    image_path = entry_image_path.get()
    if not image_path:
        messagebox.showerror("Error", "Please provide an image path.")
        return
    plot_autocorrelation(image_path)
    plot_autocorrelation(image_path.split('.')[0] + "_LogisticEnc.png")

# Create main window
root = Tk()
root.title("Image Encryption and Decryption")

# Image path
label_image_path = Label(root, text="Image Path:")
label_image_path.grid(row=0, column=0, padx=10, pady=10)
entry_image_path = Entry(root, width=50)
entry_image_path.grid(row=0, column=1, padx=10, pady=10)
button_select_image = Button(root, text="Select Image", command=select_image)
button_select_image.grid(row=0, column=2, padx=10, pady=10)

# Key
label_key = Label(root, text="Key:")
label_key.grid(row=1, column=0, padx=10, pady=10)
entry_key = Entry(root, width=50)
entry_key.grid(row=1, column=1, padx=10, pady=10)

# Buttons
button_encrypt = Button(root, text="Encrypt", command=encrypt_image)
button_encrypt.grid(row=2, column=0, padx=10, pady=10)
button_decrypt = Button(root, text="Decrypt", command=decrypt_image)
button_decrypt.grid(row=2, column=1, padx=10, pady=10)
button_histogram = Button(root, text="Plot Histogram", command=plot_histograms)
button_histogram.grid(row=3, column=0, padx=10, pady=10)
button_autocorrelation = Button(root, text="Plot Autocorrelation", command=plot_autocorrelations)
button_autocorrelation.grid(row=3, column=1, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()