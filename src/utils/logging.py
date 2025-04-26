#################################################
#########           logging           ###########
#################################################

import os
import sys
from pathlib import Path
from datetime import datetime
import csv
import pickle
import shutil

def print0(s, master_process, logfile, console=False):
    # Ensure print0 works even if not master_process (but does nothing)
    if master_process and logfile:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

def initalize_logging(master_process, cfg):
    # begin logging
    logfile = None
    experiment_dir_path = None # Define experiment_dir_path outside the if block
    metrics_csv_path = None # Define metrics_csv_path
    if master_process:
        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 1. Create the experiment directory name
        experiment_dir_name = (f"{start_time}_{cfg.model_name}")
        # 2. Create the experiment directory path
        experiment_dir_path = Path("experiments") / experiment_dir_name
        os.makedirs(experiment_dir_path, exist_ok=True)
        # 3. Set the logfile path inside the experiment directory
        logfile = experiment_dir_path / "training_log.txt"
        # 4. Set the metrics CSV file path
        metrics_csv_path = experiment_dir_path / "metrics.csv"
        print0(f"Logging to: {logfile}", master_process, logfile, console=True)
        print0(f"Metrics CSV: {metrics_csv_path}", master_process, logfile, console=True)
        # 5. Initialize metrics CSV file with headers
        with open(metrics_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step", "type", "loss", "cumulative_time_ms", "step_avg_ms"])
        # 6. Log any command-line arguments that were provided (overriding defaults)
        #cli_arg_dict = {k: v for k, v in vars(cli_cfg).items() if v is not None}
        #if cli_arg_dict:
        #    print0("Command-line arguments overriding defaults:", console=True)
        #    for key, value in cli_arg_dict.items():
        #        print0(f"  --{key} = {value}", console=True)
        #    print0("="*100, console=True)

        print0("Copying relevant files to experiment directory...", master_process, logfile)
        files_to_copy = ["requirements.txt", sys.argv[0], "download_hellaswag.py", "download_fineweb.py"]
        for file_path_str in files_to_copy:
            file_path = Path(file_path_str)
            if file_path.exists():
                try:
                    # Use Path object methods for cleaner path manipulation
                    target_path = experiment_dir_path / f"{file_path.stem}.txt"
                    shutil.copy(str(file_path), str(target_path))
                    print0(f"- Copied {file_path} to {target_path}")
                except Exception as e:
                    print0(f"- Failed to copy {file_path}: {e}", master_process, logfile)
            else:
                print0(f"- File not found, skipping: {file_path}", master_process, logfile)

        # Handle tokenizer separately as it's a .pkl file
        tokenizer_path = Path(f"data/{cfg.tokenizer}")
        if tokenizer_path.exists():
            try:
                with open(tokenizer_path, 'rb') as f:
                    tokenizer_config = pickle.load(f)
                # Save the config as a pretty-printed text file
                tokenizer_log_path = experiment_dir_path / f"{tokenizer_path.stem}_config.txt"
                import pprint
                tokenizer_str = pprint.pformat(tokenizer_config)
                with open(tokenizer_log_path, "w") as f:
                    f.write(f"Tokenizer Config ({cfg.tokenizer}):\n")
                    f.write("="*100 + "\n")
                    f.write(tokenizer_str)
                print0(f"- Saved tokenizer config to {tokenizer_log_path}", master_process, logfile)
                del tokenizer_config # Free up memory
            except Exception as e:
                print0(f"- Error processing tokenizer {tokenizer_path}: {e}", master_process, logfile)
        else:
            print0(f"- Tokenizer file not found: {tokenizer_path}", master_process, logfile)

        print0("="*100, master_process, logfile)