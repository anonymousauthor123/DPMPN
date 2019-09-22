# DPMPN
Dynamically Pruned Message Passing Networks

The code is based on our pervious naming scheme of our model. Then, we changed a lot of terms that might cause confusion when referring to components in our model in order to give a better and clearer statement in our paper. Here is the list of terminology between the previously used and the currently used:

- `unconsciousness flow` (previous): `IGNN` (now)
- `consciousness flow`: `AGNN`
- `attended nodes`: nodes in the attending-from horizon
- `seen nodes`: nodes in the attending-to horizon
- `memorized nodes`: visited nodes
- `scanned edges`: edges of neighborhood

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

