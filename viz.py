import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from PIL import Image

# Directory where images are saved
image_dir = "generated_images"
# Directory to save the animation frames
output_dir = "animation_frames"
os.makedirs(output_dir, exist_ok=True)

# Number of images
num_images = 20

# Create a figure
fig, ax = plt.subplots()

def update(frame):
    img_path = os.path.join(image_dir, f"{frame + 1}.png")
    img = mpimg.imread(img_path)
    ax.clear()
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Image {frame + 1}")

    # Save each frame as an image
    output_path = os.path.join(output_dir, f"frame_{frame + 1}.png")
    plt.savefig(output_path)

# Create animation
ani = FuncAnimation(fig, update, frames=num_images, interval=500)

# Save each frame as an image
for frame in range(num_images):
    update(frame)

# Load the frames into a list
frames = []
for frame in range(num_images):
    frame_path = os.path.join(output_dir, f"frame_{frame + 1}.png")
    frames.append(Image.open(frame_path))

# Save as an animated GIF
frames[0].save(
    'generated_numbers_animation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=500,  # Duration in milliseconds between frames
    loop=0  # Loop forever
)

print("Animation saved as generated_numbers_animation.gif")

# Optionally, display the animation
plt.show()
