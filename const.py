import platform

if platform.system() == "Windows":
    device = "cuda:0"

elif platform.system() == "Linux":
    device = "cuda:0"




elif platform.system() == "Darwin":
    device = "mps:0"

else:
    raise ValueError("Unknown platform: {}".format(platform.system()))

# device = "cpu"