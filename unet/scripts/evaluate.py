from simple_converge.main.main import test
from simple_converge.tf_models.models_collection import models_collection

from unet.Settings import Settings
from unet.Dataset import Dataset

# Create class instances
settings = Settings()
dataset = Dataset()

# Test
test(settings, dataset, models_collection)
