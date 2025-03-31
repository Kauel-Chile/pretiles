import numpy as np

def imagenp_pad_img_multiplo(img,multiplo=64):
    """ Agrega filas y columnas con zeros a una imagen hasta que sus dimensiones sean multiplos del parámetro multiplo
    img: Numpy Array float32 con la forma (h,w,c).
    retorna: Numpy array float32 con la forma (h+deltah,w+deltaw,c).
    """
    h,w = img.shape
    th = int(max(np.ceil(h/multiplo)*multiplo,1.0))
    tw = int(max(np.ceil(w/multiplo),1.0)*multiplo)
    deltah=th-h
    deltaw=tw-w
    if(deltaw>0):
        cols=np.zeros((h,deltaw),dtype=img.dtype)
        img=np.concatenate([img,cols],axis=1)
    if(deltah>0):
        rows=np.zeros((deltah,tw),dtype=img.dtype)
        img=np.concatenate([img,rows],axis=0)
    return img

def create_input(img):
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 1, 2))
    return img

def pad_multiplo(img, mask):
    img = imagenp_pad_img_multiplo(img, 32)
    mask = imagenp_pad_img_multiplo(img, 32)
    return img, mask     