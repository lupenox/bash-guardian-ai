import yaml


def load_config(path="training/lora_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    config = load_config()
    print("Bash Guardian AI LoRA Training Scaffold")
    print("Model:", config["model_name"])
    print("Dataset:", config["dataset_path"])
    print("Output:", config["output_dir"])
    print("Future training code will be implemented here.")
