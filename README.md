# Multi-Paradigm-MAP-Elites-Quality-Diversity-Neural-Architecture-Generation-with-DARTS-cells

NAG-ME-QD
Neural Architecture Generation with MAP-Elites Quality-Diversity Optimization
An independent research project that automatically discovers diverse, high-performing neural network architectures using evolutionary MAP-Elites optimization, zero-shot evaluation proxies, and multi-paradigm search across convolutional, transformer, and recurrent operations.

Author: Pratheeksha Aravind
Status: Work in progress — targeting NeurIPS 2026 workshops


What it does
Instead of designing a single best architecture, NAG-ME-QD maintains an archive of diverse high-performing architectures, each occupying a unique region of a 32-dimensional behavior space. This means the system discovers many creative solutions rather than converging on one.
The core loop:

Generate candidate architectures (random init or mutation of existing elites)
Evaluate fitness via zero-shot proxies (no full training needed)
Place into the MAP-Elites archive based on behavioral fingerprint
Repeat, with elites used as parents for future mutations

Top architectures from the archive are then fully trained for final evaluation.

Results
Experiment run: 20 MAP-Elites iterations, zero-shot evaluation mode, CIFAR-10
ModelArchitectureVal AccuracyParamsTop-1 (overall)Hybrid (conv-dominant)54.4%40,842Top-2 (overall)Hybrid (conv-dominant)56.1%56,330Best TransformerHybrid (transformer-dominant)58.0%57,482Best RecurrentDAG (recurrent-dominant)44.3%77,194
Archive: 24 elites discovered, QD score 11.99, max fitness 0.612

Note: Results likely reflect basic CNN structure and training hyperparameters more than the NAS components. See Limitations in the paper.

Experiment data and trained weights are available on Hugging Face.

File structure
nag-meqd/
├── README.md
├── LICENSE
├── .gitignore
│
├── NAG_MEQD_Complete_SelfContained.ipynb   ← start here
│
├── nag_mapelites_qd.py                     # Core: architecture specs, cells, operations, training
├── nag_mapelites_qd_part2.py               # Core: evaluator, mutator, MAP-Elites grid
└── nag_behavior_space_32d.py               # 32D behavior space, ResultsTracker, SparseArchiveGrid

Quickstart (Google Colab)
The recommended way to run this is via the included Colab notebook, which handles setup, Google Drive checkpointing, and result downloads automatically.

Upload NAG_MEQD_Complete_SelfContained.ipynb to Google Colab
Run the setup cell (installs dependencies)
Configure iterations, eval mode, and dataset in the config cell
Run


Local setup
bashgit clone https://github.com/[YOUR_USERNAME]/nag-meqd.git
cd nag-meqd
pip install torch torchvision numpy tqdm
python nag_mapelites_qd_main_updated.py
Requires Python 3.8+, PyTorch 1.12+. GPU recommended but not required.

Architecture search space
Three cell types:

DAG cells — directed acyclic graph with flexible topology
DARTS cells — differentiable architecture search cells
Hybrid cells — mix of the above

Three operation types across cells:

Convolutional (standard convs, depthwise, dilated)
Transformer (multi-head attention, feedforward)
Recurrent (LSTM, GRU)

Behavior space: 32 dimensions characterizing each architecture across topology, operation mix, parameter count, depth, skip connections, and more.

Citation
If you use this work, please cite:
Pratheeksha Aravind. "Self-Generating Neural Networks: NAG-ME-QD: Neural Architecture 
Generation with MAP-Elites Quality-Diversity Optimization." Unpublished manuscript, 2026.

License
MIT
