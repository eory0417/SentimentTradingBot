import argparse
import json
import signal
import subprocess
import sys
from pathlib import Path

def merge_config(config_path: Path, secrets_path: Path, merged_path: Path, disable_telegram: bool = False) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    with secrets_path.open("r", encoding="utf-8") as f:
        secrets = json.load(f)

    exchange = config.get("exchange", {})
    exchange["key"] = secrets.get("binance", {}).get("apiKey", "")
    exchange["secret"] = secrets.get("binance", {}).get("secret", "")
    config["exchange"] = exchange

    telegram_secrets = secrets.get("telegram", {})
    if disable_telegram:
        config.pop("telegram", None)
    else:
        telegram_config = config.get("telegram", {})
        if telegram_secrets.get("token") and telegram_secrets.get("chat_id"):
            telegram_config["enabled"] = True
            telegram_config["token"] = telegram_secrets["token"]
            telegram_config["chat_id"] = telegram_secrets["chat_id"]
            config["telegram"] = telegram_config
        elif "telegram" in config:
            config["telegram"] = telegram_config

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    if disable_telegram:
        print(f"Telegram polling disabled in merged config: {merged_path}")
    else:
        print(f"Merged config saved with Telegram settings: {merged_path}")

    return merged_path


def clean_args(raw_args):
    cleaned = []
    skip_next = False
    for arg in raw_args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--config", "-c", "--secrets"):
            skip_next = True
            continue
        if arg == "--disable-telegram":
            continue
        cleaned.append(arg)
    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Run freqtrade with secrets merged from user_data/secrets.json into a temporary config file."
    )
    parser.add_argument(
        "--config",
        default="user_data/config.json",
        help="Path to the base Freqtrade config file."
    )
    parser.add_argument(
        "--secrets",
        default="user_data/secrets.json",
        help="Path to the secrets JSON file."
    )
    parser.add_argument(
        "--disable-telegram",
        action="store_true",
        help="Merge the config with secrets but disable Telegram polling to avoid getUpdates conflicts."
    )
    parser.add_argument(
        "freqtrade_args",
        nargs=argparse.REMAINDER,
        help="Remaining arguments to pass to freqtrade, e.g. trade --dry-run --freqaimodel PyTorchMLPRegressor",
    )
    args = parser.parse_args()

    if not args.freqtrade_args:
        parser.error("Missing freqtrade command. Example: trade --dry-run --freqaimodel PyTorchMLPRegressor")

    freqtrade_args = args.freqtrade_args
    if freqtrade_args[0] == "--":
        freqtrade_args = freqtrade_args[1:]
    freqtrade_args = clean_args(freqtrade_args)

    base_config_path = Path(args.config)
    secrets_path = Path(args.secrets)
    merged_config_path = Path("user_data") / "config_merged_with_secrets.json"

    merge_config(base_config_path, secrets_path, merged_config_path, disable_telegram=args.disable_telegram)

    command = [sys.executable, "-m", "freqtrade"] + freqtrade_args + ["--config", str(merged_config_path)]
    process = subprocess.Popen(command)

    def cleanup_and_exit(signum, frame):
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass
        try:
            merged_config_path.unlink()
        except OSError:
            pass
        sys.exit(130)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    try:
        result = process.wait()
    finally:
        try:
            merged_config_path.unlink()
        except OSError:
            pass

    sys.exit(result)


if __name__ == "__main__":
    main()
