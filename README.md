# [CSCLNet: Cross-temporal and spatial information fusion for multi-task building change detection using multi-temporal optical imagery](https://authors.elsevier.com/sd/article/S1569-8432(24)00429-1)

Our work has been accepted by JAG.

**CSCLNet** is a multi-task network that detects 2D and 3D urban changes from optical images, setting new benchmarks on 3DCD and SMARS datasets.
## Installation

- Download [Python 3](https://www.python.org/)
- Install the packages:
```bash
pip install -r requirements.txt
```

## Usage 

To train the model, prepare a *.yaml* file and put in the ```config``` directory and then run the following command:
```bash
python train.py --config="your_config_file"
```
To test your model, follow the same steps for training and then run the following command:
```bash
python test.py --config="your_config_file"
```


## Related resources

In this section, we point out some useful repositories, resources and connected projects. 

- [BIT Github repository](https://github.com/justchenhao/BIT_CD) 
- [3DCD Github repository](https://github.com/VMarsocci/3DCD)

## Citation

If you found our work useful, consider citing our work.


## License

Code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.

