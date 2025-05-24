# ATGen: Active Text Generation

## How to launch without config

```bash
HYDRA_CONFIG_NAME=base HYDRA_CONFIG_PATH=./../../../configs python3 src/atgen/run_scripts/run_active_learning.py al.query_size=10 al.num_iterations=5 a
l.query_size=10 data.dataset=Harvard/gigaword data.input_column_name=document data.output_column_name=summary labeler.type=golden al.strategy=random
```