import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18

from tqdm import tqdm

from utils import *
# hyper
features = ['SOH','voltage_measured', 'current_measured',
            'temperature_measured', 'time']
batch_size = 1  # 1*len(every_file)
input_size = len(features)
hidden_size = 128
num_layers = 1
output_size = 1
seq_len = 20   # 预测序列长度
epoch = 1000   # 1*len(train_directory)
learning_rate = 0.001  # upgrade to adaptive lr?

save_path = 'seq{}_.pth'.format(str(seq_len))  # model path
train_directory = '../datasets/train/'
val_directory = '../datasets/val/'
test_directory = '../datasets/alldataset/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir',
    default=test_directory,
    help='Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation')
parser.add_argument(
    '--model_dir',
    default="./model/float_model/",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path: https://download.pytorch.org/models/resnet18-5c106cde.pth'
)
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--inspect', 
    dest='inspect',
    action='store_true',
    help='inspect model')

parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()



def evaluate(model, val_loader, states):

  model.eval()
  model = model.to(device)
  with torch.no_grad():
      for test_data, test_data_real in val_loader:
          test_data = torch.squeeze(test_data).to(device)
          test_data_real = torch.squeeze(test_data_real).to(device)
          test_output = model(test_data,states)
          test_output = torch.squeeze(test_output)

          print('---------------------------------')
          RMSE = torch.sqrt(torch.mean((test_output - test_data_real)**2))
          print('RMSE:{}'.format(RMSE))
  return RMSE


def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

  data_dir = args.data_dir
  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  inspect = args.inspect
  config_file = args.config_file
  target = args.target
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers).cpu()
  model.load_state_dict(torch.load(file_path, map_location=torch.device(device)))

  input_x = torch.zeros([batch_size, seq_len, input_size])
  input_states = ((torch.zeros(num_layers, batch_size, hidden_size),torch.zeros(num_layers,batch_size,hidden_size)),
                  (torch.zeros(num_layers, batch_size, hidden_size),torch.zeros(num_layers,batch_size,hidden_size)),
                  (torch.zeros(num_layers, batch_size, hidden_size),torch.zeros(num_layers,batch_size,hidden_size)),
                  (torch.zeros(num_layers, batch_size, hidden_size),torch.zeros(num_layers,batch_size,hidden_size)))
  
  if quant_mode == 'float':
    quant_model = model
    if inspect:
      if not target:
          raise RuntimeError("A target should be specified for inspector.")
      import sys
      from pytorch_nndct.apis import Inspector
      # create inspector
      inspector = Inspector(target)  # by name
      # start to inspect
      inspector.inspect(quant_model, (input_x,input_states), device=device)
      sys.exit()
      
  else:
    ####################################################################################
    # This function call will create a quantizer object and setup it. 
    # Eager mode model code will be converted to graph model. 
    # Quantization is not done here if it needs calibration.
    quantizer = torch_quantizer(quant_mode = quant_mode, 
                                module = model,
                                # input_args = (input_x,),
                                input_args = (input_x,input_states),
                                device=device,
                                quant_config_file=config_file,
                                target=target,
                                bitwidth = 16,
                                lstm = True) 

    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  loss_fn = torch.nn.MSELoss().to(device)


  val_dataset = LoadDataset(val_directory, seq_len=seq_len, features=features)
  val_loader  = DataLoader(val_dataset, batch_size=batch_size)

  # fast finetune model or load finetuned parameter before test
  if finetune == True:
      ft_dataset = LoadDataset(val_directory, seq_len=seq_len, features=features)
      ft_loader  = DataLoader(ft_dataset, batch_size=batch_size)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, input_states))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
  evaluate(quant_model, val_loader, input_states) 


  # handle quantization result
  if quant_mode == 'calib':
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_torch_script()
    quantizer.export_onnx_model()
    quantizer.export_xmodel()


if __name__ == '__main__':

  model_name = 'B_seq20_420'
  file_path = os.path.join(args.model_dir, model_name + '.pth')

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path)

  print("-------- End of {} test ".format(model_name))