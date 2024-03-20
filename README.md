# LUT-DM-filters
Starting with the implementation outlined in [Diffpose](https://github.com/GONGJIA0208/Diffpose), we streamlined the model architecture by removing the context encoder and GMM components and we customized it to handle only 12 human keypoints.


## MICRO-BENCHMARKING
Look at `lut_analysis.ipynb`

## LUT-based real-time filtering
Download the dataset of Total Capture [here](https://cvssp.org/data/totalcapture/).

Apply noise to the ground truth dataset, creating a hierarchical structure as depicted below:

    data
    └── total_capture
        ├── noisy
        └── gt

### LUT-DM / SMART-DM
To run the LUT-DM or SMART-DM or ORACLE scripts, use the following command-line format:
```bash
python <file_name> --path <path> --gt_path <gt_path> 
```
#### Where:
- `<file_name>`: Name of the Python script to execute.
    - Options: `lut_launch_name.py`, `smart_launch_name.py`, `oracle_bench_name.py`
- `<path>`: path to the folder containing noisy data.
- `<gt_path>`: path to the folder containing ground truth data.

##
To run the DDPM or DDIM scripts, use the following command-line format:
```bash
python ddpm_name.py --path <path> --gt_path <gt_path> --mode <mode> 
```
#### Where:
- `<path>`: path to the folder containing noisy data.
- `<gt_path>`: path to the folder containing ground truth data.
- `<mode>`: Type of noise to apply to the dataset.
    - Options: `ddpm`, `ddim`

Examples:
```bash
python ddpm_name.py --path data/total_capture/noisy/ --gt_path data/total_capture/gt/ --mode ddpm
```
##

Note: In our implementation we use `common` and `models` folder from [Diffpose](https://github.com/GONGJIA0208/Diffpose)
