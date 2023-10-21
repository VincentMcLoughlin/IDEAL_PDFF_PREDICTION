from DataManager import DataManager
from models.Resnet50 import Resnet50
def main():

    # config_path = "config.yaml"
    # print("main")
    # data_manager = DataManager(config_path)
    # train_dataset, test_dataset, val_dataset = data_manager.build_datasets(batch_size=32)

    model = Resnet50(64, num_classes=10)

if __name__ == "__main__":
    main()