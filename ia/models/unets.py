import io
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
from torch import nn
from zipfile import ZipFile

INPLACE=False 
CONVPAD='zeros'

def pt_get_activation(activation) -> nn.Module:
    """ Retorna un modulo de activacion por nombre.
    Retorna None si el nombre no coincide """
    if(activation is None): return None
    elif isinstance(activation,nn.Module): return activation
    elif(activation=='relu'): return nn.ReLU(inplace=INPLACE)
    elif(activation=='elu'): return nn.ELU(inplace=INPLACE)
    elif(activation=='leakyrelu'): return nn.LeakyReLU(inplace=INPLACE)
    elif(activation=='sigmoid'): return nn.Sigmoid()
    elif(activation=='logsigmoid'): return nn.LogSigmoid()
    elif(activation=='softmax'): return nn.Softmax(dim=1)
    elif(activation=='softmin'): return nn.Softmin(dim=1)
    elif(activation=='logsoftmax'): return nn.LogSoftmax(dim=1)
    elif(activation=='prelu'): return nn.PReLU()
    elif(activation=='relu6'): return nn.ReLU6(inplace=INPLACE)
    elif(activation=='rrelu'): return nn.RReLU(inplace=INPLACE)
    elif(activation=='selu'): return nn.SELU(inplace=INPLACE)
    elif(activation=='celu'): return nn.CELU(inplace=INPLACE)
    elif(activation=='gelu'): return nn.GELU(approximate='tanh')
    elif(activation=='silu'): return nn.SiLU(inplace=INPLACE)
    elif(activation=='mish'): return nn.Mish(inplace=INPLACE)
    elif(activation=='softplus'): return nn.Softplus()
    elif(activation=='softsign'): return nn.Softsign()
    elif(activation=='softshrink'): return nn.Softshrink()
    elif(activation=='tanh'): return nn.Tanh()
    return None

class Conv(nn.Module):
    """ Convolution + Activation """
        
    def __init__(self, ic:int, oc:int, k=3, s=1, p=1, bias=True, activation=None, scale=None, residual=False):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, kernel_size=k, stride=s, padding=p, bias=bias, padding_mode=CONVPAD)
        self.activation = pt_get_activation(activation)
        self.scale = scale
        self.residual = residual
        
    def forward(self, x):
        x0 = x
        x = self.conv(x)
        if(self.activation is not None): x = self.activation(x)
        if(self.scale is not None): x = self.scale(x)
        if(self.residual): x = x0 + x
        return x
    
class ResID07(nn.Module):
    """ Bloque residual bn+act+conv+bn+act+conv con ajuste de dimensiones espaciales y semanticas """ 
    def __init__(self, ic:int, oc:int, activation='relu', dropout=0.0, expansion=1, resample=None):
        super().__init__()
        mc = int(ic*expansion)
        self.norm1 = torch.nn.BatchNorm2d(ic, momentum=0.01)
        self.act1  = pt_get_activation(activation) 
        self.conv1 = Conv(ic,mc,k=3,s=1,p=1)
        
        self.resample = resample
        
        self.norm2 = torch.nn.BatchNorm2d(mc, momentum=0.01)
        self.act2  = pt_get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout>0.0 else None
        self.conv2 = Conv(mc,oc,k=3,s=1,p=1)
        
        self.conv3 = Conv(ic,oc,k=1,s=1,p=0) if ic!=oc else None
        
    def forward(self, x, emb=None):
        x0 = x
        if(self.norm1 is not None): x = self.norm1(x)
        if(self.act1  is not None): x = self.act1(x)
        x = self.conv1(x)
        
        if(emb is not None): x = x+emb
        
        if(self.resample is not None):
            x  = F.interpolate(x, scale_factor=self.resample, mode='bilinear')
            x0 = F.interpolate(x0,scale_factor=self.resample, mode='bilinear')
        
        if(self.norm2   is not None): x = self.norm2(x)
        if(self.act2    is not None): x = self.act2(x)
        if(self.dropout is not None): x = self.dropout(x)
        x = self.conv2(x)
        
        if self.conv3 is not None: x0 = self.conv3(x0) #(b,oc,h,w) Ajusta los canales para que sean iguales.
        
        return x0 + x
    
class ResID07N(nn.Module):
    """ N bloques ResID07 """
    def __init__(self, ic:int, n=2, activation='relu', dropout=0.0, expansion=2):
        super().__init__()
        self.layers = []
        for _ in range(n): self.layers.append(ResID07(ic,ic, activation=activation, dropout=dropout, expansion=expansion))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

class UNetSam(nn.Module):
    """ UNET
    La primera seccion aplica ResID07 con resample=0.5 varias veces para reducir  la resolucion y aumentar la cantidad de filtros, hasta llegar a la maxima cantidad de filtros.
    La segunda seccion aplica ResID07 con resample=2.0 varias veces para aumentar la resolucion y reducir  la cantidad de filtros.
    Agrega enlaces entre la primera y segunda seccion.
    Retorna una lista 'y' con todas las salidas de cada nivel, siendo y[0] la entrada e y[-1] la ultima salida
    activation2: Activacion de SAM: sigmoid, softmax8, softmax16, softmax32
    """
    
    def __init__(self, filters=(1,16,32,64,128,256,128,64,32,16), n=1, activation='relu', dropout=0.0, expansion=1):
        super().__init__()
        c = len(filters) #Total de filtros
        j = np.argmax(filters) #Indice de la mayor cantidad de filtros (Parte mas ancha de la UNet)
        
        self.convs = nn.ModuleList() #Lista de modulos ConvBnAct
        self.convs_res = nn.ModuleList() #Lista de modulos ResID01N

        for i in range(1,j+1):
            f1 = filters[i-1]
            f2 = filters[i]
            self.convs.append( ResID07(f1,f2,resample=0.5,activation=activation, dropout=dropout, expansion=expansion))
            module = ResID07N(f2,n=n,activation=activation, dropout=dropout, expansion=expansion)
            self.convs_res.append(module)
            
        self.convts = nn.ModuleList() #Lista de modulos ConvTBnAct (Convolucion Transpuesta)
        self.convts_res = nn.ModuleList() #Lista de modulos ResID01N
        self.links = nn.ModuleList() #Lista de modulos ConvBnAct para enlazar la primera seccion con la segunda
        for i in range(j+1,c):
            f1 = filters[i-1]
            f2 = filters[i]
            self.convts.append(ResID07(f1,f2,resample=2,activation=activation, dropout=dropout, expansion=expansion))
            module = ResID07N(f2,n=n,activation=activation, dropout=dropout, expansion=expansion)
            self.convts_res.append(module)
            self.links.append(ResID07N(f2,n=n,activation=activation, dropout=dropout, expansion=expansion))
            
        self.c = c #Total de filtros
        self.j = j #Indice de la mayor cantidad de filtros

    def forward(self, x):
        """ x:(b,c,h,w) """

        y = [x] #Lista de salidas. El primer valor de salida es la entrada
        for conv,res  in zip(self.convs, self.convs_res): #Por cada convolucion de la primera seccion
            x = conv(x) #Aplica la convolucion
            
            x = res(x) #Aplica residual
            y.append(x) #Guarda la salida
            
        i=1
        j=self.j
        for convt,res,link in zip(self.convts, self.convts_res, self.links): #Por cada convolucion transpuesta y enlace
            x = convt(x) #Aplica la convolucion transpuesta 
            
            x = res(x) #Aplica residual
            if(j-i>=0): x=x+link(y[j-i]) #Aplica el enlace a la primera seccion
            i+=1
            y.append(x) #Guarda la salida
        
        return y #Retorna una lista con todas las salidas
    

class Unet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.unet = UNetSam(filters=(1,16,32,64,128,256,128,64,32,16), n=1, activation='relu', dropout=0.0, expansion=1)
        self.convT_o1 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convT_o2 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        

    def forward(self, x):
        x -=  F.avg_pool2d(x, 101, stride=1, padding=50)
        y = self.unet(x)
        y1 = self.convT_o1(y[-1])
        output_pretil = self.convT_o2(y[-1])
        output_class = F.softmax(y1, dim=1)        
        return output_class, output_pretil

