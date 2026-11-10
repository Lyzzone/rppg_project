import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Configuration-driven rPPG training and inference")
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("train", "evaluate", "predict"):
        command = commands.add_parser(action)
        command.add_argument("--config", type=Path, required=True)
        if action != "train":
            command.add_argument("--checkpoint", type=Path, required=True)
    plot = commands.add_parser("visualize")
    plot.add_argument("--predictions", type=Path, required=True)
    plot.add_argument("--video-id", help="Exact video_id from the prediction CSV")
    args = parser.parse_args()

    try:
        if args.action == "visualize":
            from .visualize import visualize

            for path in visualize(args.predictions, args.video_id):
                print(path.resolve())
            return

        import torch

        from .config import load_config

        config_path = args.config.resolve()
        cfg = load_config(config_path)
        runtime = cfg["runtime"]
        torch.set_num_threads(runtime["num_threads"])
        torch.manual_seed(runtime["seed"])
        name = runtime["device"]
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        if name != "cpu" and not (name == "cuda" or (name.startswith("cuda:") and name[5:].isdigit())):
            raise ValueError("runtime.device must be cpu, auto, cuda or cuda:N")
        if name.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; set runtime.device = 'cpu'")
        device = torch.device(name)

        if args.action == "train":
            from .train import train
            from .visualize import plot_training_loss

            test_after_train = cfg["train"]["test_after_train"]
            if test_after_train and "test" not in cfg["data"]:
                raise ValueError("train.test_after_train requires [data.test]")
            run = train(cfg, config_path, device)
            print(f"training_run={run}", flush=True)
            test_metrics = None
            try:
                if test_after_train:
                    from .evaluate import evaluate

                    test_run = evaluate(cfg, config_path, run / "best.pt", device,
                                        output_dir=run / "test")
                    test_metrics = json.loads((test_run / "metrics.json").read_text(encoding="utf-8"))
                    print(f"test_loss={test_metrics['loss']:.4f} best_epoch={test_metrics['checkpoint_epoch']}")
            finally:
                print(f"loss_plot={plot_training_loss(run / 'training.csv', test_metrics)}")
        elif args.action == "evaluate":
            from .evaluate import evaluate

            run = evaluate(cfg, config_path, args.checkpoint.resolve(), device)
        else:
            from .predict import predict

            run = predict(cfg, config_path, args.checkpoint.resolve(), device)
        print(f"run_dir={run}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
