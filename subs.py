import subprocess

def run_conda_script(conda_init_path, env_name, script_path):
    # Source the Conda initialization script
    subprocess.run(f"source {conda_init_path}", shell=True, check=True)

    # Activate Conda environment
    subprocess.run(f"conda activate {env_name}", shell=True, check=True)

    # Run the Python script within the Conda environment
    subprocess.run(f"python {script_path}", shell=True, check=True)

if __name__ == "__main__":
    # Path to the Conda initialization script (e.g., /path/to/miniconda3/etc/profile.d/conda.sh)
    conda_init_path = "/path/to/miniconda3/etc/profile.d/conda.sh"

    # Name of the Conda environment to activate
    env_name = "deepspray_segmentation"

    # Path to the Python script to run within the Conda environment
    script_path = "~deepspray_segmentation/dpplus-segmentation/train_for_segmentation.py train --weight=coco --backbone=resnet50 --epoch=10"

    run_conda_script(conda_init_path, env_name, script_path)
