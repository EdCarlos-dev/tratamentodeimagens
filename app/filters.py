# colocar todos os filtros 

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numba import jit

#@jit(nopython = True)
class Filters:

    def __init__(self, path):
        self.image = cv2.imread(path)
        
    
    def image_to_gray(self, image):
        image_gray = np.zeros((len(image), len(image[0])),dtype=np.uint8)

        for i in range(len(image)):
            for j in range(len(image[0])):
                image_gray[i][j] = (int( image[i][j][0] )+ int(image[i][j][1] )+ int(image[i][j][2] ) )/ len(image[0][0])
        
        return image_gray

   
    def image_threshold(self, image_gray, limiar):
    
        input_limiar = np.zeros((len(image_gray), len(image_gray[0])),dtype=np.uint8)

        for i in range(len(image_gray)):
            for j in range(len(image_gray[0])):

                aux = image_gray[i][j]
                if aux <= limiar:
                    input_limiar[i][j] = 0
                else:
                    input_limiar[i][j] = 255
                    
        return input_limiar

    def zero_box(self, image, kernel=3):

        # lado =  7  kernel
        var = int((kernel - 1) / 2)
        mask_ones = image
        mask_zeros = np.zeros((len(mask_ones) + kernel - 1, len(mask_ones[0]) + kernel - 1), dtype=np.uint8) # cria a borda de zeros
        mask_zeros[var:len(mask_ones) + var, var:len(mask_ones[0]) + var] = mask_ones # sobrepoe a imagem sentralizada na matriz

        return mask_zeros

  
    def out_box_zero(self, image_in_box, kernel=3):
        
        var = int((kernel - 1) / 2)
        image_out_box = image_in_box[var:len(image_in_box) - var, var:len(image_in_box[0]) - var]

        return image_out_box


   
    def medias_kernel(self, kernel, image_gray, x=0, y=0):
        '''
        dado um kernel
        calcula a média dos valores dos pixels e retorna o valor  
        '''
        var = int((kernel-1)/2)
        x = x-var
        y = y-var
        media = 0

        for i in range(x, x + kernel):
            
            for j in range (y, y + kernel):
            
                media+=image_gray[i][j] # somando os valores das coordenadas

        result = media/(kernel*kernel) # fazendo a média dos valores
        

        return result
    


    def threshoud_adapt(self, image_gray, kernel, constante_c):

        input_thresh = self.zero_box(image_gray, kernel)
        matriz_suporte = np.zeros((len(input_thresh), len(input_thresh[0])),dtype=np.uint8)
        borda = int((kernel-1)/2) 
    
        for i in range(len(input_thresh)): #linhas 
            for j in range(len(input_thresh[0])): #colunas
                
                if (i >= borda and i < (len(input_thresh) -borda))  and ( j >= borda and j < len(input_thresh[0]) -borda):
                    media_trash = self.medias_kernel(kernel,input_thresh,i,j) - constante_c

                    if input_thresh[i][j]< media_trash:
                        matriz_suporte[i][j] = 0
                    else:
                        matriz_suporte[i][j] = 255

        res = self.out_box_zero(matriz_suporte, kernel)
        
        return res


    def histogram_weight(self, lista):
        
        # Set minimum value to infinity
        final_min = np.inf
        # total pixels in an image
        total = np.sum(lista[0])
        for i in range(256):
            # Split regions based on threshold
            left, right = np.hsplit(lista[0],[i])
            # Splt intensity values based on threshold
            left_bins, right_bins = np.hsplit(lista[1],[i])
            # Only perform thresholding if neither side empty
            if np.sum(left) !=0 and np.sum(right) !=0:
                # Calculate weights on left and right sides
                w_0 = np.sum(left)/total
                w_1 = np.sum(right)/total
                # Calculate the mean for both sides
                mean_0 = np.dot(left,left_bins)/np.sum(left)
                mean_1 = np.dot(right,right_bins[:-1])/np.sum(right)  # right_bins[:-1] because matplotlib has uses 1 bin extra
                # Calculate variance of both sides
                var_0 = np.dot(((left_bins-mean_0)**2),left)/np.sum(left)
                var_1 = np.dot(((right_bins[:-1]-mean_1)**2),right)/np.sum(right)
                # Calculate final within class variance
                final = w_0*var_0 + w_1*var_1
                # if variance minimum, update it
                if final<final_min:
                    final_min = final
                    thresh = i

        return thresh


    def otsu(self, image_gray):
        lista = plt.hist(image_gray.ravel(),256,[0,256])
        plt.clf()
        limiar_o = self.histogram_weight(lista)
        return self.image_threshold(image_gray , limiar_o)



    def erode_kernel(self, kernel,imagem, x=0, y=0):
        '''
        dado um kernel
        calcula a média dos valores dos pixels e retorna o valor  
        '''
        var = int((kernel-1)/2)
        x = x-var
        y = y-var
        white = 0
        black = 0

        for i in range(x, x + kernel):
            for j in range (y, y + kernel):
                if imagem[i][j] == 255:
                    white += 1
                else:
                    black += 1
                
        if white == kernel*kernel:
            return 255

        else:
            return 0
        
