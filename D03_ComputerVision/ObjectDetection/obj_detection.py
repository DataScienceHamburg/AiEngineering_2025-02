#%% packages
from ultralytics import YOLO

# %%
# %%
model = YOLO("yolo11n.pt")
# %%
# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)
# %%
