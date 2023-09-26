from DataManager import DataManager

def main():

    config_path = "config.yaml"
    print("main")
    data_manager = DataManager(config_path)
    train_dataset, test_dataset, val_dataset = data_manager.build_datasets(batch_size=32)

if __name__ == "__main__":
    main()