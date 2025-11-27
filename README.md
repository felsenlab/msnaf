## Installation

## Usage

## Design -- Not A Pipeline
This codebase is explicitly Not A Pipeline - instead, it's designed around modules, which allow users to write their own pipeline using any combination of modules. This is a more flexible approach than linear systems that require branching and can be difficult to extend or modify.
The idea is that users can add modules as new functionality is required without having to modify pipelines and cdoe paths and program logic. 

### Modules

Modules are atomic, that is, they perform a single task. Much like a function, which should 'DO' only one thing, or a class, which should encapsulate a single 'BEHAVIOR', modules should perform one step in a data processing pipeline.
For example, these are good modules: 
 - processCameraVideo
 - timestampEphys
 - trainSaccadeClassifier
 - 

and these are bad modules:
 - loadEphysAndLabjackAndCombineAndAlignTimestamps

Modules are data-oriented - the inputs should be data arrays, and the output should be data arrays. Modules should not modify existing data in-place (i.e., modules should create new datasets, not modify existing ones).
Modules should take file paths or dataset paths and functions should take and output numpy arrays, lists, dicts, or dataframes.  NOTE: these are not hard and fast rules -- if you have a really good reason (such as: ) you can violate these guidelines. 

```python
@Module(
    name="foo",
    requires=["normalized_data"],  # datasets
    produces=["foo_metrics"]
)
```

Modules specify three things: 1) The name of the module, 2) Required inputs in the form of a list of named fields, and 3) the output of the module in the form of a list of named fields.



#### Module registry


#### Adding a new modules
-- extend basemodule --> must implement some functions


### Pipelines
Pipelines are a collection of modules that get run in order. They are defined by YAML config files, which contain a list of modules and parameters. Example pipelines are shown below:

```yaml
pipeline:
  - module: "load_raw_data"
    config:
      path: "input.csv"

  - module: "normalize"
    config:
      center: true
      zscore: true

  - module: "summaries"
    config: {}pipeline:
```

#### Dependency resolution

#### Skip/Overwrite

#### Output validation

#### Loading existing pipelines

#### Creating new pipelines

#### Provenance

