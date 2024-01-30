# Griavty Monitoring design optimazation
## Usage
Run the following code:
'''
python gravity_monitoring.py
'''

# Sensitivity-Based Seismic Monitoring Design

Seismic sources and receivers are deployed at locations with high sensitivity to the target of interest.

## Usage
1. Download and install Anaconda on your computer
2. Create a python virtual environment using the following command:

```
conda env create -f environment.yml
```

3. Activate the virtual environment depending on which operating system and command shell you’re using.

- On Unix or MacOS, using the bash shell: source /path/to/venv/bin/activate
- On Unix or MacOS, using the csh shell: source /path/to/venv/bin/activate.csh
- On Windows using the Command Prompt: path\to\venv\Scripts\activate.bat

4. Run the monitoring design script using Python:

```
python sensitivity_optimization.py [--config path_to_custom_yaml_file.yaml]
```

### Parameters

- `--config`: Path to the configuration YAML file. If not provided, the default `seis_sens_opt_params.yaml` will be used.
- A configuration file contains parameters of seismic simulation setup, optimization settings and choices of 
  output data and images.

### Output
- An output directory will be created in the same directory of this Python script
- The output directory consists of several subdirs for time-lapse CO2 plume models, baseline velocity models, 
  sensitivity images, and optimal seismic monitoring design
- More information can be found in the RAMP design document.
