from main.main import test
from tf_models.models_collection import models_collection

from generator.Generator import Generator
from unet.Settings import Settings
from unet.Dataset import Dataset

# Create class instances
settings = Settings()
dataset = Dataset()
generator = Generator()

# Test
test(settings, dataset, generator, models_collection)
