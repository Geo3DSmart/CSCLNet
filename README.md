# [CSCLNet: Cross-temporal and spatial information fusion for multi-task building change detection using multi-temporal optical imagery](https://authors.elsevier.com/sd/article/S1569-8432(24)00429-1)

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
```XML
@article{xiao2024cross,
  title={Cross-temporal and spatial information fusion for multi-task building change detection using multi-temporal optical imagery},
  author={Xiao, Wen and Cao, Hui and Lei, Yuqi and Zhu, Qiqi and Chen, Nengcheng},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={132},
  pages={104075},
  year={2024},
  publisher={Elsevier}
}

@article{xiao20233d,
  title={3D urban object change detection from aerial and terrestrial point clouds: A review},
  author={Xiao, Wen and Cao, Hui and Tang, Miao and Zhang, Zhenchao and Chen, Nengcheng},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={118},
  pages={103258},
  year={2023},
  publisher={Elsevier}
}
```

## License

Code is released for non-commercial and research purposes only. For commercial purposes, please contact the authors.

