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

