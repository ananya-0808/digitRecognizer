# Handwritten Digit Recognizer

A neural network that recognizes handwritten digits, with a desktop app where you
draw a digit and see the model's prediction along with a bar chart of its
confidence for all 10 digits.

## Features

- Trains a neural network on the MNIST dataset (about 97% test accuracy)
- Drawing pad built with Tkinter for testing your own handwriting
- Bar chart showing the model's probability for each digit, 0 through 9
- Preprocessing that crops, scales, and centers your drawing to match the MNIST format

## How it works

1. `train.py` loads the MNIST dataset (70,000 images of handwritten digits),
   trains a small neural network with TensorFlow/Keras, and saves it as
   `digit_reader.keras`.
2. `handwritten_digits.py` loads the saved model and opens a drawing window.
   When you click **Predict**, your drawing is shrunk to 28x28 pixels, normalized,
   and passed to the model, which outputs a probability for each digit.

The model has a Flatten layer, two Dense layers of 128 neurons with ReLU, and a
10-neuron softmax output layer.

## Getting started

### Requirements

- Python 3
- The packages in `requirements.txt`

### Installation

```
git clone https://github.com/ananya-0808/digitRecognizer.git
cd digitRecognizer
pip install -r requirements.txt
```

### Run it

Train the model first (this only needs to be done once):

```
python3 train.py
```

Then open the drawing app:

```
python3 handwritten_digits.py
```

Draw a digit on the white pad, then click **Predict**. Click **Clear** to start over.

## What I learned

- The model scored well on the test set but struggled on my own drawings until I preprocessed them to match how
  MNIST images are formatted (cropped, scaled to 20x20, and centered).
- It often confuses 1 up the most, if not centered perfectly

## Possible improvements

- Replace the Dense layers with a convolutional neural network (CNN) for better
  accuracy on messy handwriting

## Credits

Built with reference to the GeeksforGeeks tutorial
[Classify Handwritten Digits with TensorFlow](https://www.geeksforgeeks.org/python/python-classifying-handwritten-digits-with-tensorflow/).
The drawing app, probability chart, input preprocessing, and styling are my own additions.
