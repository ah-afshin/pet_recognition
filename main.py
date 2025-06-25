import torch as t

from src.models import PetBreedsRecogPreTrainedMobileNetV3
from src.data_setup import get_breeds_dataloader
from src.train import train_fine_tune_breeds_recog_model, train_features_extracted_breeds_recog_model
from src.test import test_breeds_recog_model

# Hyperparameters
from config import LEARNING_RATE, EPOCHS, FINE_TUNE_AT_EPOCH, FINE_TUNING_DEPTH


def main() -> None:
    """Main training pipeline for pet recognition model.

    Args:
        None
    
    Returns:
        None
    """
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    train_dl, test_dl = get_breeds_dataloader()

    model = PetBreedsRecogPreTrainedMobileNetV3().to(device)
    train_fine_tune_breeds_recog_model(
        model, train_dl,
        epochs=EPOCHS,
        device=device,
        learning_rate=LEARNING_RATE,
        fine_tune_at_epoch=FINE_TUNE_AT_EPOCH,
        depth=FINE_TUNING_DEPTH
    )
    test_breeds_recog_model(model, test_dl, device=device)
    print("accuracy on traindata:"); test_breeds_recog_model(model, train_dl, device=device)

    model_name = f"model_18__breedsrecog_pretrained_featureextract_mobilenet_v3"
    t.save(model.state_dict(), "models/"+model_name+".pth")



if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n\texecution time: {execution_time}s\n")
