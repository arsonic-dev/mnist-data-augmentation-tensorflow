from model import make_model
from dataset import load_datasets

# Load datasets
augmented_train, non_augmented_train, validation = load_datasets()

# Train model without augmentation
model_without_aug = make_model()
print("Training model WITHOUT data augmentation...")
no_aug_history = model_without_aug.fit(
    non_augmented_train,
    epochs=50,
    validation_data=validation
)

# Train model with augmentation
model_with_aug = make_model()
print("Training model WITH data augmentation...")
aug_history = model_with_aug.fit(
    augmented_train,
    epochs=50,
    validation_data=validation
)
