# discrete Walk-Jump Sampling (dWJS)

This is the official open source repository for [discrete Walk-Jump Sampling](TODO) developed by [Prescient Design, a Genentech accelerator.](https://gene.com/prescient)


## Notes


## Setup and Usage
### Environment
We used the following GPU-enabled setup with `conda` (originally run in an HPC environment with NVIDIA A100 GPUs).
```

```


### Model weights
PyTorch model weights and hyper-parameter configs for the models trained on antibody datasets as described in the manuscript are stored in `models` directory.

### Run sampling
To sample using a trained model, users can run the following scripts:

```
$ python...
```
## Contributing

We welcome contributions. If you would like to submit pull requests, please make sure you base your pull requests off the latest version of the `main` branch. Keep your fork synced by setting its upstream remote to `Genentech/walk-jump` and running:

```sh
# If your branch only has commits from master but is outdated:

$ git pull --ff-only upstream main


# If your branch is outdated and has diverged from main branch:

$ git pull --rebase upstream main
```

## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


## Citations
If you use the code and/or model, please cite:
```
```