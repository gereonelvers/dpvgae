## Incorporating Dynamic Graphs into Graph Neural Networks for Business Processes Redesign and Concept Drift Prediction
- Generation of dataset is done using `process_dataset_generator.py`
- `bpmn_dataset.py` is the dataset class
  - Note that for now, the total number of graphs in the dataset must be set manually in the dataset during the process model import
- `sampling.py` allows for faster manual sampling of the edge threshold by saving and then reimporting predicted adjacency matrices
- Model and training loop can be found in
    - `previous_iterations/process_gcnn.py`
    - `previous_iterations/process_vgae.py`
    - `directed_process_vgae.py` + `layers.py`  
   for the three model iterations
- Dataset and logs will be written to `./data-dump` folder
- Visualize loss tensorboard by running `tensorboard --logdir ./data-dump/logs`

Other folders:
- `./literature-review` contains the literature corpus and review script
- `./legacy` Contains small experiments and old code, neither of which are documented in detail. Included more for completeness than anything else.


### In order to ensure that are running the most recent version of the code, check out the GitHub repository [here](github.com/gereonelvers/dpvgae).