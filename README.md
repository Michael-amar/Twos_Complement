# Twos Complement
Artifact Repository for the paper "Two's Complement: Monitoring Software Control Flow Using Both Power and Electromagnetic Side Channels”, by Michael Amar, Yossi Oren, Lojenaa Navanesan and Asanka Sayakkara, to be presented at the 27th Euromicro Conference on Digital System Design.

# Running the notebooks
The Multimodal folder contains a notebook for any one out of the three integration approaches that were presented in the paper.
To prepare the environment to run them:

1. Create the environment:
```bash
conda create --name testenv python=3.8.15
conda activate testenv
python -m pip install -r requirements.txt
```


2. Download and extract the test dataset
```bash
gdown 1cUMKuaC5HvfSDBHM3I-ubhnkb8oYQOpY
tar -xvzf testset.tar.gz
```

3. Download and extract the pretrained models
```bash
gdown 1Nfov-AWuc7Aouy_-tEljGrRP3uT1A4oz
tar -xvzf trained_models.tar.gz
```

4. Run any of the notebooks in the Multimodal folder

# For More Information
Please see the [paper's official webpage](https://orenlab.sise.bgu.ac.il/p/TwosComplement)
