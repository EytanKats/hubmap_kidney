from simple_converge.main.main import inference
from simple_converge.tf_models.models_collection import models_collection

from unet.Settings import Settings
from unet.Dataset import Dataset

# Create class instances
settings = Settings()
dataset = Dataset()

# Run inference
inference(settings, dataset, models_collection)
