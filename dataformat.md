# header
- training_model_file_name: str
- LUPIN_version: str
- runtime_sec: float

# images
- id: int
- dataset_id: int
- path: str
- width: int
- height: int
- file_name: str

# categories
- id: int
- name: str

# annotations
- image_id: int
- category_id: int
- segmentation: [polygon]
- bbox: [left_top_x, left_top_y, right_bottom_x, right_bottom_y]
- score: float
