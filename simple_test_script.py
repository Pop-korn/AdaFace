import inference
from PIL import Image

image_path = 'data/style_images/a.jpg'

model = inference.load_pretrained_model()

img = Image.open(image_path).convert('RGB')
bgr_input = inference.to_input(img)

feature, _ = model(bgr_input)

print(feature)
