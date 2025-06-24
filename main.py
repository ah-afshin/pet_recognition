import torch as t

from src.models import PetBreedsRecognitionMobileNet_v2
from src.data_setup import get_breeds_dataloader
from src.train import train_breeds_recog_model
from src.test import test_breeds_recog_model

# Hyperparameters
from config import LEARNING_RATE, EPOCHS#, WEIGHT_DECAY, SCHEDULE_LR_STEP, SCHEDULE_LR_GAMMA, FEATURE_MAP, HIDDEN_UNITS


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

    model = PetBreedsRecognitionMobileNet_v2().to(device)
    # model.load_state_dict(t.load(f="models/model_11__breedsrecog_MobileNet_v1.pth"))
    # print("\nmodel is loaded")
    # test_breeds_recog_model(model, test_dl, device=device)
    # print("\ncontinue training...\n")
    
    train_breeds_recog_model(
        model, train_dl,
        epochs=EPOCHS,
        device=device,
        learning_rate=LEARNING_RATE,
        # weight_decay=WEIGHT_DECAY,
        # schedule_lr_step=SCHEDULE_LR_STEP,
        # schedule_lr_gamma=SCHEDULE_LR_GAMMA
    )
    test_breeds_recog_model(model, test_dl, device=device)
    print("accuracy on traindata:"); test_breeds_recog_model(model, train_dl, device=device)

    model_name = f"model_12__breedsrecog_MobileNet_v2"
    t.save(model.state_dict(), "models/"+model_name+".pth")



if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n\texecution time: {execution_time}s\n")
