# IFCDR

This is the official implementation code for **IFCDR**.

## Requirements

Please ensure the following packages are installed before running the code:

* torch >= 3.6
* numpy
* pandas
* tqdm
* scipy
* scikit-learn
* pillow
* lmdb
* keras

## Usage

**Step 1: Single-Domain Training**

Train the single-domain recommendation model by running:
```bash
python main_single_mf.py
```
**Step 2: Configuration**

Modify the config.json file to select the cross-domain training settings and set the initialization parameters.

**Step 3: Train Cross-Domain Model**

Finally, train the cross-domain recommendation model by running:
```bash
python python main_cross.py
```



