import base64

image_path = "/mnt/data/HarmonyBatch/HD-wallpaper-black-and-white-boring-outline-simple.jpg"  # Replace with your image file
output_file = "image_base64.txt"

# Read and encode the image in Base64
with open(image_path, "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Save Base64 string to a file
with open(output_file, "w") as base64_file:
    base64_file.write(img_base64)

print(f"Base64 string saved to {output_file}")
