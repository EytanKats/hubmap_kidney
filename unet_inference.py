from main.main import inference
from tf_models.models_collection import models_collection

from generator.Generator import Generator
from unet.Settings import Settings
from unet.Dataset import Dataset

# Create class instances
settings = Settings()
dataset = Dataset()
generator = Generator()

# Run inference
inference(settings, dataset, generator, models_collection)
