import argparse
from xgboost import XGBRegressor
from scripts.predict_target import predict_target
from scripts.save_model import save_model

"""
Main CLI for DemandPredictor project. You can write only one string to run scripts. Just copy one of the line below to the terminal.

Usage examples:
    python main.py predict --model_folder saved_models/xgb_model
    python main.py train --data_path data/raw_data.csv --target Deals --models_folder_path saved_models --model_name xgb_model
"""

def run_predict(model_folder: str):
    result = predict_target(model_folder=model_folder)
    print(f"Predicted deals for next month: {int(result)}")

def run_train(data_path: str, target: str, models_folder_path: str, model_name: str):
    model = XGBRegressor()
    save_path = save_model(
        model=model,
        data_path=data_path,
        target=target,
        models_folder_path=models_folder_path,
        model_name=model_name
    )
    print(f"Model trained and saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DemandPredictor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict next month's deals")
    predict_parser.add_argument("--model_folder", type=str, required=True, help="Path to saved model folder")

    train_parser = subparsers.add_parser("train", help="Train and save model")
    train_parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    train_parser.add_argument("--target", type=str, required=True, help="Target column name")
    train_parser.add_argument("--models_folder_path", type=str, required=True, help="Folder to save models")
    train_parser.add_argument("--model_name", type=str, required=True, help="Model name to save")

    args = parser.parse_args()

    if args.command == "predict":
        run_predict(args.model_folder)
    elif args.command == "train":
        run_train(args.data_path, args.target, args.models_folder_path, args.model_name)