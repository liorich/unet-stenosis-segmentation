# Deciphering an angiographic photograph of the coronary arteries by deep learning

### Authors: Lior Ichiely, Noam Yaakobi

## Introduction

Angiographic coronary imaging is a valuable technique used to assess the condition and health of arteries. This diagnostic method involves the injection of iodine into the patient's arteries, followed by X-ray imaging to examine the flow of this contrast agent. In this project, the narrowing of coronary arteries was discussed, known as Stenosis.

Nowadays, this segmentation of the narrowing within the images is done manually and carefully by a physician. This project proposed utilizing Convolution Neural Networks (CNNs) to expedite and enhance the segmentation process. Several scientific papers suggested that for medical image segmentation tasks the UNet++ CNN Architecture performs with high accuracy.

## Dependencies
```
Python 3.10.9
tensorflow-gpu>2.4
keras>2.4
numpy
matplotlib
scikit-image
tqdm
Nvidia CUDA 11.2, Nvidia CUDnn 8.7
Robustness of Probabilistic U-Net github repository, includes the UNet++ and Attention UNet++ models. 
https://github.com/rizalmaulanaa/Robustness_of_Prob_U_Net
```

## Training

Training parameters we used:
```
Weight initialization: Adam
Kernel Size: 3x3
Dataset size and split: 164 images, 131 for training, 13 for validation and 30 for testing
Loss function: Binary Cross-entropy
Batch size: UNet++ - 4, AttUNet++ - 5
Number of Epochs: 80
Evaluation Metrics: Number of trainable parameters, Intersection Over Union (IoU), Coordinate Hits and False Alarms.```

The research process originally involved the acquisition of angiographic data, manual segmentation, training of both networks, performance comparison to select the best network, statistical analysis of the data and finally, creating a GUI for comfortable use of the entire system. Because the acquired dataset does not include any personal information, statistical analysis could not be extracted.
The dataset used is publicly available on **Mendeley Data**. It includes a set of angiographic imaging series of one hundred patients who underwent coronary angiography at the Research Institute for Complex Problems of Cardiovascular Diseases. There are over 7,000 images that include the video frames of the iodine injection from different angles to different patients. The dataset also includes the coordinates of detected stenosis for each frame. In total, 164 images were selected and manually colored according to the given coordinates. The selection involved a variety of angles and patients to avoid overfitting the networks.

To expedite and enhance the detection process, a recent study proposed the implementation of Convolutional Neural Network (CNN) architectures. Based on previous research, two neural network architectures were selected, and their performance were compared – UNet++ versus AttUNet++. Both network architectures are evolutions of the basic encoder-decoder architecture, UNet.

![A frame with detected Stenosis. A visual representation of the coordinates on the left, and the manual coloring on the right.](/assets/readme/segmentation.jpg)

Both networks detected stenosis correctly in at least 80% of the test images. AttUNet++ detected more images than UNet++ but conversely, raised more false alarms. In conclusion, the Attention mechanism’s extra complexity improved detection but could reduce segmentation accuracy. For further research, it would be recommended to use segmented, more accurate and up-to-date data.

![Coloring of two angiographic images by the two networks, in both methods.](/assets/readme/tests.jpg)

## Conclusion

Our results indicate that AttUNet++ is more suited for the medical task at hand.
In terms of average IoU, both networks performed almost equally and therefor the metric of Coordinate Hits was given more weight. Another reason for this consideration is the importance of detecting the stenosis over correctly coloring it in the images. AttUNet++ got more coordinate hits than UNet++ in both coloring methods.
