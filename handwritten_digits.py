import tkinter as tk #makes the canvas, window and buttons
import numpy as np #handles the arrays of numbers
import tensorflow as tf #builds and trains the neural networks
from PIL import Image, ImageDraw

model = tf.keras.models.load_model("digit_reader.keras") #rebuilds the trained model so it doesn't have to be trained every time

SIZE = 280 #width and height of drawing pad
BRUSH = 10 #brush size

class DigitApp: #runs once automatically when the app is created
    def __init__(self,root): #holds everything about the app
        root.title("Draw a digit")
        self.canvas = tk.Canvas(root, width = SIZE, height = SIZE, bg = "black")
        self.canvas.pack() #pack places widget in the window
        self.image = Image.new("L", (SIZE, SIZE), 0) #makes a new image
        self.pen = ImageDraw.Draw(self.image) #creates a pen for drawing on that image
        self.canvas.bind("<B1-Motion>", self.paint) #when the left click button is held down, self.paint is calles
        tk.Button(root, text = "Predict", command = self.predict).pack(side = "left", padx = 10, pady = 10) #creates the predict button
        tk.Button(root, text = "Clear", command = self.clear).pack(side = "left") #creates the clear button
        self.result = tk.Label(root, text = "Draw a digit, then click 'Predict'", font = ("Arial", 14)) #explains what to do
        self.result.pack(pady = 10)
        self.chart = tk.Canvas(root, width = SIZE, height = 150, bg = "white") #creates another canvas for the other bar charts
        self.chart.pack(pady = 10)
        self.draw_chart(np.zeros(10)) #passes ten zeros to see the layout before predicting

    def paint(self, event): #runs while the mouse is dragged
        x, y = event.x, event.y #mouse's position on the canvas
        box = (x - BRUSH, y - BRUSH, x + BRUSH, y + BRUSH)
        self.canvas.create_oval(box, fill = "white", outline = "white")
        self.pen.ellipse(box, fill = 255) #makes the dots look like smooth strokes

    def draw_chart(self, probs): #model's confidence for digits 0 to 9
        self.chart.delete("all") #erases everything on the chart so old bars don't pile up under new ones
        bar_width = SIZE / 10 #sets up the layout
        max_height = 100
        baseline = 120
        best = int(np.argmax(probs))
        for i, p in enumerate(probs): #runs 10 times, once per digit
            x0 = i * bar_width + 5 #left edge
            x1 = (i + 1) * bar_width - 5 #right edge
            y0 = baseline - p * max_height #top of the bar
            color = "tomato" if i == best and probs.max() > 0 else "steelblue" #winning digit gets a red bar, the rest get blue bars
            self.chart.create_rectangle(x0, y0, x1, baseline, fill = color, outline = "") #draws the bar from its top-left corner
            self.chart.create_text((x0 + x1) / 2, baseline + 15, text = str(i))
            self.chart.create_text((x0 + x1) / 2, y0 - 8, text = f"{p:.0%}", font = ("Arial", 8)) #centers the text

    def preprocess(self):
        bbox = self.image.getbbox()          # box around the drawn pixels
        if bbox is None:
            return None                      # nothing drawn
        img = self.image.crop(bbox)

    # scale so the longest side is 20 pixels, keeping the shape
        w, h = img.size
        scale = 20 / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))

    # paste into the middle of a 28x28 black image
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
        arr = np.array(canvas, dtype="float32")

    # shift so the center of mass of the ink is at the center
        ys, xs = np.indices(arr.shape)
        total = arr.sum()
        cy = (ys * arr).sum() / total
        cx = (xs * arr).sum() / total
        arr = np.roll(arr, (round(13.5 - cy), round(13.5 - cx)), axis=(0, 1))
        return arr

    def predict(self):
        arr = self.preprocess()
        if arr is None:
            self.result.config(text="Draw something first!")
            return
        arr = arr.reshape(1, 28, 28)
        arr = tf.keras.utils.normalize(arr, axis=1)
        probs = model.predict(arr, verbose=0)[0]
        digit = int(np.argmax(probs))
        self.result.config(text=f"Prediction: {digit}  ({probs[digit]:.0%} sure)")
        self.draw_chart(probs)

    def clear(self): #resets everything
        self.canvas.delete("all")
        self.pen.rectangle((0, 0, SIZE, SIZE), fill = 0)
        self.result.config(text = "Draw a digit, then click Predict")
        self.draw_chart(np.zeros(10))

root = tk.Tk() #creates the main window
DigitApp(root) #builds the app inside it
root.mainloop() #starts the event loop



