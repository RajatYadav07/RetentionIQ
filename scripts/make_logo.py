from PIL import Image, ImageDraw
import os

# Create a transparent image 256x256
img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Draw a professional blue/cyan geometric shape (e.g. an isometric cube or network nodes)
# Colors: Blue: #3b82f6, Cyan: #06b6d4, Light Blue: #60a5fa
cyan = (6, 182, 212, 255)
blue = (59, 130, 246, 255)
light_blue = (96, 165, 250, 255)

# Network-like connected nodes and a cube
# Nodes
n1 = (128, 40)
n2 = (40, 90)
n3 = (216, 90)
n4 = (128, 140)
n5 = (40, 200)
n6 = (216, 200)
n7 = (128, 250)

# Lines
draw.line([n1, n2], fill=light_blue, width=16)
draw.line([n1, n3], fill=light_blue, width=16)
draw.line([n2, n4], fill=blue, width=16)
draw.line([n3, n4], fill=blue, width=16)
draw.line([n2, n5], fill=blue, width=16)
draw.line([n3, n6], fill=blue, width=16)
draw.line([n4, n7], fill=cyan, width=16)
draw.line([n5, n7], fill=cyan, width=16)
draw.line([n6, n7], fill=cyan, width=16)

# Draw circles at nodes
def node(center, r, color):
    draw.ellipse([center[0]-r, center[1]-r, center[0]+r, center[1]+r], fill=color)

r = 18
node(n1, r, light_blue)
node(n2, r, light_blue)
node(n3, r, light_blue)
node(n4, r, blue)
node(n5, r, blue)
node(n6, r, blue)
node(n7, r, cyan)

assets_dir = r"d:\Data Science project\RetentionIQ\app\dashboard\assets"
os.makedirs(assets_dir, exist_ok=True)
img.save(os.path.join(assets_dir, "retentioniq_logo.png"), "PNG")
img.save(os.path.join(assets_dir, "retentioniq_favicon.png"), "PNG")

print("Generated logos successfully.")
