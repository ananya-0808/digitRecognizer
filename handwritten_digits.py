import tkinter as tk #makes the canvas, window and buttons
import numpy as np #handles the arrays of numbers
import tensorflow as tf #builds and trains the neural networks
from PIL import Image, ImageDraw

model = tf.keras.models.load_model("digit_reader.keras") #loads 
