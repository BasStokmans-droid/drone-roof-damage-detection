# Thesis listings 5.2 and 5.3 (Codefragment): generating EigenCAM heatmaps for a trained YOLOv8 model
# with the open-source YOLOv8_Explainer package. Paths are placeholders, as in the thesis.
from YOLOv8_Explainer import yolov8_heatmap, display_images

model = yolov8_heatmap(
    weight="/location/model.pt",
    conf_threshold=0.4,
    method="EigenCAM",
    layer=[10, 12, 14, 16, 18, -3],
    ratio=0.02,
    show_box=True,
    renormalize=False,
)

imagelist = model(img_path="/location/image.jpg")
display_images(imagelist)
