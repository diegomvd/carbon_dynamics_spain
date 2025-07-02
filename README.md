# Recent climate change reduced Spanish forests' carbon sink capacity

Repository contains all the code needed to reproduce the analysis presented in the manuscript **Recent climate change reduced Spanish forests' carbon sink capacity**. The analysis pipeline can be used to infer woody biomass measurements from satellite imagery, analyse spatio-temporal patterns and relationship with recent climatic anomalies. 

The repository is organized around multiple components, each representing a modelling step. Within each component, core algorithmic logic modules are located within the **core/** directories, and wrapper scripts to run different pipeline stages are found below the **scripts/** directories. Each component has an associated configuration file that centralizes parametrization. 

Conda environment files needed to execute different parts of the pipeline are located within the **environments** directory at the repository root. 

The most convenient way of running the pipeline and make modifications is to install the codebase as a package in editable mode by running: _pip install -e ._ in repository root. The code can be used by running the dedicated scripts in each component or by executing pre-made recipes (in repository root) that orchestrate and chain a series of these scripts to execute multiple steps at once. 

Most of the computing is too demanding for a single-machine and cannot be done in a personal laptop. We executed the pipeline in a local cluster of 5 workstations with 24 cores and 192 Gb of RAM each. The canopy height modelling component trains Neural Networks and requires access to GPUs. 

ALS height data can be downloaded from: https://pnoa.ign.es/web/portal/pnoa-lidar/presentacion
National can be downloaded forest inventory data from: https://www.miteco.gob.es/es/biodiversidad/temas/inventarios-nacionales/inventario-forestal-nacional/cuarto_inventario.html

For any questions please contact me at: diego.bengochea@mncn.csic.es
