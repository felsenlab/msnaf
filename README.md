## Installation
1. Install conda
2. Clone the git repo (`git clone https://github.com/felsenlab/msnaf`)
3. Create the conda repo: `conda env create --file msnaf/environment.yaml`
4. Activate conda environment: `conda activate deeplabcut-tensorflow`
5. Install MSNAF: `pip install -e .`

Note that the `pip install -e .` installs the package as editable. That means that any changes to the repository are immediately reflected without reinstalling the package. This also means that only ONE version of the package exists on the system. Never again do you have to navigate to `/home/user/anaconda3/envs/deeplabcut-tensorflow/lib/python3.10/site-packages/flimsy` to edit a file!

# Usage
### Run Pipeline
`flimsy --pipeline pipeline_config.yaml --run session_list.yaml`

### Get Modules Information:
`flimsy --module_help loadLabJackDataRig2.0`

### List Modules (NOT IMPLEMENTED):
`flimsy --list_modules`

## Design -- Not A Pipeline
This codebase is explicitly Not A Pipeline - instead, it's designed around modules, which allow users to write their own pipeline using any combination of modules. This is a more flexible approach than linear systems that require branching and can be difficult to extend or modify. The idea is that users can add modules as new functionality is required without having to modify pipelines, code paths, or program logic. 

### Modules

Modules are atomic, that is, they perform a single task. Much like a function, which should 'DO' only one thing, or a class, which should encapsulate a single 'BEHAVIOR', modules should perform one step in a data processing pipeline.
For example, these are good modules: 
 - timestampFrames
 - loadPoseData
 - runDLCOnVideos
 - trainSaccadeClassifier

and these are bad modules:
 - loadEphysAndLabjackAndCombineAndAlignTimestamps
 - processVideos (ambiguous at best, too many things at worst)

#### Module Design

Modules are data-oriented - the inputs should be data (arrays, lists, dicts, strings, ints, floats, etc), and the output should be data. Modules should not modify existing data in-place (i.e., modules should create new datasets, not modify existing ones).
Modules can load files, but should not modify existing files or create new files. They should ONLY return new data for the pipeline to save into the designated output file. NOTE: these are not hard and fast rules -- if you have a really REALLY good reason (such as: the deeplabcut module calls `dlc.AnalyzeVideos()`, which creates the deeplabcut pose files) you can violate these guidelines. 

#### Module Metadata

Modules are just functions, but we use a python feature called Decorators to add additional metadata to them so the pipeline knows what to do with them. Decorators are just functions, and you call them just like you would any other function. 

The first decorator is the `@module` tag. This declares the module name and a description. When the module is imported, the decorator labels the run function with the name and desription and adds the module adds the module to a central registry (discussed in the next step).

The next decorator is the `@requires` tag. This declares that the module needs data contained in the `"Foo"` field of results file. When the pipeline gets to this module in the pipeline, it will load this data and pass it to the module.

The next decorator is the `@param` tag. Similar to `@requires`, this decorator tells the pipeline that the module needs this data to run. In contrast to the @requires decorator though, @params MUST be provided by the user, either in the pipeline configuration file or the run configuration file (see below for sections on those files). Optionally, a default value can be set, which is used if an overriding value is not provided in the pipeline config or run config. 

The last decorator is `@produces`. This decorator tells the pipeline what data the module will return, and what to call it. The pipeline will save the data produced by the module to the field name provided. 

```python
@module(name="identifyCandidateSaccades", description="Identifies candidate events based on threshold of horizontal eye velocity")
@requires("pose/smoothed/right", description="Smoothed eye position data")
@param("velocity_percentile", description="Percentile used during peak detection", default=95)
@produces("candidateSaccadeIndices", description="Indices of detected events")
def run():
    return 
```

<!-- Modules specify three things: 1) The name of the module, 2) Required inputs in the form of a list of named fields, and 3) the output of the module in the form of a list of named fields. -->


#### Placeholders & Wildcards

** NOT CURRENTLY IMPLEMENTED ** 

Sometimes a module may not know beforehand EXACTLY what it will produce, but it knows the general format. In this case, you can use the following syntax to tell the pipeline that multiple things matching the pattern will be produced by the module. 

```python
@produces("/pose/{bodypart}/raw")
def load_pose(...):
    ...
```
This says:
>“This module will produce one or more concrete datasets, each corresponding to a specific value of bodypart.”


#### Module registry
A list of available modules is maintained by the pipeline. When the pipeline is started, it begins by loading all of the files in the module directory. Each module is registered in a central repository. This repository can be queried to obtain 1) a list of available modules, 2) detailed information about a module, and 3) find modules which produce a desired data field.

 `flimsy --list_modules`

 `flimsy --module_help identifyCandidateSaccades`

 `flimsy --produces saccades/predicted/waveforms`

#### Adding a new modules

Adding modules is reasonably straightforward. As explained above, modules are functions that are decorated with various metadata (`@module`, `@requires`, `@produces`, and `@param`). You can add a new module using the template below. Note that a module MUST declare a name and at least one @produces field. The pipeline calls each module as `module_fn(data, params)`, where all the `@requires` fields are provided in `data` as a `dict`, and all the `@params` are provided in `params` as a `dict`. These can be accessed inside the module as `data[<fieldname>]`, where `fieldname` is the name of the param or requries field. 

```python
from flimsy.pipeline.basemodule import * ## initalizes components required for all modules

logger = logging.getLogger(__name__) ## enables you to print information to the terminal/console and a log file

@module(name="Foo")
@produces("Bar")
def mymodule(data, params):
    return 
```

### Logging
Every source file in the repository implements a python logger. This logger prints to the console and saves a log file. Python loggers enable granular control over what gets printed to the console. If you are writing a new module you may want detailed information about errors to help with debugging. In this case you might call `flimsy --loglevel debug`, which will print all `logger.debug()` statements to the console and log file. Once the module is working though, all that printing might be distracting. Now you can set loglevel to `info`, `warning`, or even `error`, so that only urgent messages about things going wrong are printed. 

### Pipelines
Pipelines are a collection of modules that get run in order. They are defined by YAML config files, which contain a list of modules and parameters. Example pipelines are shown below. To add a module to the pipeline, simply add it's name to the list. Modules with user definable params (e.g., any @param declarations) can be specified using the colon, indent, param_name pattern seen in the template. Modules without params should be set to {} (an empty dict).

The pipeline steps through the list of modules one at a time: first it loads the data declared in the `@requires` and `@params` decorators and passes this data to the module as `data` and `params`. It then executes the module and saves the output to the output file. 

```yaml
initH5:
  metadata_pattern: "metadata.txt"
loadLabJackDataRig2.0:
    file_pattern: "labjack_data.csv"
    channel_map:
      AIN1: "stimulus"
      FIO6: "camera"
      FI06: "camera"
timestampCameraFrames:
  sampling_rate: 2000
  interval_pattern: "*Cam_timestamps.txt"
example_module: {}
```

#### Loading existing pipelines
Run configuration and pipeline configuration files are purposely separated to reduce how much you have to write to analyze new sessions. The idea is that you can write a single pipeline file for a set of experiments, and when you have new sessions to analyze, all you have to do is provide a run config file with a list of paths to the new session folders. 

### Run Configuration Files
Run configuration files are similar in structure to pipeline configuration files (see example below). Like pipeline configuration files, they consist of module names and their associated parameters. However, unlike pipeline configuration files, which must declare all of the modules that need to be run, run configuration files only need to declare the file creation module and the list of session folders to analyze. In addition, any parameters specifc to those files can also be provided. For example, if the pipeline configuration file sets the default value for `sampling_rate` as 2000, but for these sessions the sampling rate was 1000, you can override that default by declaring the module name and the sampling rate param.

```yaml
initH5:
  basepath:
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session005/
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session006/
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session007/
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session008/
```

```yaml
initH5:
  basepath:
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session005/
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session006/
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session007/
    - /home/schollab-dion/Documents/felsen_pipeline/garcian/session008/
timestampCameraFrames:
    sampling_rate: 1000
```

 #### Creating new pipelines
Creating new pipelines is as simple as creating a .yaml file. You can put it anywhere, but there's a special folder dedicated to pipeline configs in the root directory of the package (~/Code/msnaf).


#### Provenance

** NOT IMPLEMENTED **

The pipeline will eventually save metadata about: when the output data was produced, the version of the pipeline that generated it, etc. This information is often necessary when looking back at data that was analyzed a long time ago. It can also be used in the overwrite logic described below.  


#### Dependency resolution

** NOT IMPLEMENTED **

The pipeline will eventually provide tools for validating pipelines before running them. This process will check all of the modules to make sure that each module's @required are generated by the modules that come before it. It will also check to make sure that the pipeline doesn't call modules that don't exist, etc. 

#### Skip/Overwrite

** NOT IMPLEMENTED **

The plan is to allow per-module control over whether to overwrite previous data. This may be useful in reducing the time it takes to run the pipeline, especially during debugging of new modules. It will use a combination of user settings and provenance to determine if a module should be re-run or to use the existing data.

#### Output validation

** NOT IMPLEMENTED **

The pipeline is designed to support automatic validaton of output data. Instead of users having to write their own validation for each module, there will be pipeline wide validators which can be called to ensure that output data doesn't contain NaNs, Infs, or other common data issues. 




