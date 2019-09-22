# NeuCFlow
Neural Consciousness Flow

## Training and Evaluating

```bash
./run.sh --dataset <Dataset>
```

<Dataset> can be one of 'FB237', 'FB237_v2', 'FB15K', 'WN18RR', 'WN18RR_v2', 'WN', 'YAGO310', 'NELL995'.

## Visualization

Run with `test_output_attention` to get data files of extracted subgraphs for each query in test. For example, if you want to get data files on the NELL995 dataset (containing several separate datasets), with `max_attended_nodes=20`, run:

```bash
./run.sh --dataset NELL995 --test_output_attention --max_attended_nodes 20 --test_max_attended_nodes 20
```

Then, you get data files in the `output/NELL995_subgraph` directory. Next, visualize them by:

```bash
cd code
python visualize.py --dataset NELL995
```

You will find image files for visualization in the `visual/NELL995_subgraph` directory.

