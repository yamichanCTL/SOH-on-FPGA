# SOH-on-FPGA

## Get Start

```shell
pip install -r requirement
```

## Unzip Dataset

Download [battery dataset](https://ti.arc.nasa.gov/c/5) and delete B0038 B0039 B0040 B0041

![](.//img//SOH.jpg)

Split the dataset into train set, val set,  test set.(8:1:1 or other ratio)


project/  
│    
├── datasets/  
│  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;├── train/  
│		&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── .mat  
│  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;└── val/  
│		&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── .mat  
│  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;└── test/  
│	   	&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── .mat  
│  
├── LSTM/  
└── README.md  
## Run LSTMtest.ipynb in Order

## Run Quant
cd Vitis-AI  
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest  
conda activate vitis-ai-pytorch  
cd SOH-on-FPGA/LSTM  

python quantify.py --quant_mode calib --subset_len 1  
python quantify.py --quant_mode test --subset_len 1 --batch_size=1 --deploy  
