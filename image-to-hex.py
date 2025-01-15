with open("/mnt/data/HarmonyBatch/HD-wallpaper-black-and-white-boring-outline-simple.jpg", "rb") as image_file:
    hex_data = image_file.read().hex()
    print(hex_data)
