from numba import jit
import os    
import pandas as pd   
import matplotlib.pyplot as plt   
import numpy as np
import matplotlib.image as image 

@jit(nopython = True)

def zero_box(image, kernel=3):

    '''
    colocando a imagem centralizada em uma matriz de zeros , 
    fazendo uma borda
    '''
    var = int((kernel - 1) / 2)
    mask_ones = image
    mask_zeros = np.zeros((len(mask_ones) + kernel - 1, len(mask_ones[0]) + kernel - 1), dtype=np.uint8) # cria a borda de zeros
    mask_zeros[var:len(mask_ones) + var, var:len(mask_ones[0]) + var] = mask_ones # sobrepoe a imagem sentralizada na matriz

    return mask_zeros

def calc_box_image(image_in_box, kernel):

    var = int((kernel - 1) / 2)
    image_calculated_in_box = np.zeros((len(image_in_box), len(image_in_box[0])),dtype=np.uint8)
    
    for i in range(var, len(image_in_box)-var): #linhas 

        for j in range(var, len(image_in_box[0])-var): #colunas
            
           # if (i > var and i < (len(image_in_box) -var))  and ( j > var and j < len(image_in_box[0]) -var):
                #                                    ATUAL                ANTERIOR         SUPERIOR ESQUERDO              SUPERIOR             SUPERIOR DIREITO         POSTERIOR               INFERIOR DIREITO                  INFERIOR               INFERIOR ESQUERDO     
            image_calculated_in_box[i][j] = ((int(image_in_box[i][j])+ int(image_in_box[i][j-1])+ int(image_in_box[i-1][j-1])+ int(image_in_box[i-1][j])+ int(image_in_box[i-1][j+1])+ int(image_in_box[i][j+1])+ int(image_in_box[i+1][j+1])+ int(image_in_box[i+1][j] )+ int(image_in_box[i+1][j-1] ))/9)

    return image_calculated_in_box

def medias_kernel(kernel,imagem, x=0, y=0):
   
    '''
    gerando a "matriz mascara"
    para calcumo dos pízels adjascentes ao pixel alvo 
    Carregando na variável media a soma dos valores nos pixels
    '''

    var = int((kernel-1)/2)
    x = x-var
    y = y-var
    media = 0
    for i in range(x, x + kernel):
        for j in range (y, y + kernel):
            media+=imagem[i][j] # somando os valores das coordenadas

    result = media/(kernel*kernel) # fazendo a média dos valores
    
    return result

def out_box_zero(image_in_box, kernel=3):

    '''retirando as bordas da imagem'''
    var = int((kernel - 1) / 2)
    image_out_box = image_in_box[var:len(image_in_box) - var, var:len(image_in_box[0]) - var]

    return image_out_box

k = 5 # kernel
l = int((k-1)/2) # borda
picture = input_gray

m_box_zeros = zero_box(picture) # matriz de zeros maior que a imagem 
m_box_image = m_box_zeros.copy()
m_box_image[l:len(picture) + l, l:len(picture[0]) + l] = picture # centralizando a imagem


for i in range(len(m_box_image)):
    for j in range(len(m_box_image[0])):
            
        if (i >= l and i < (len(m_box_zeros) -l))  and ( j >= l and j < len(m_box_zeros[0]) -l):
            media = medias_kernel(k, m_box_image,i,j)
            m_box_zeros[i][j] = media

m_box_zeros = out_box_zero(m_box_zeros, 5)
m_box_image = out_box_zero(m_box_image, 5)
