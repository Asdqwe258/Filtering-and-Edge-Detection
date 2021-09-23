#Do not import any additional modules
import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt

### Load, convert to grayscale, plot, and resave an image
I = np.array(open('Iribe.jpg').convert('L'))/255
plt.imshow(I,cmap='gray')
plt.axis('off')
plt.show()

plt.imsave('test.png',I,cmap='gray')

### Part 1
def gausskernel(sigma):
    #Create a 3*sigma x 3*sigma 2D Gaussian kernel
    sum = 0
    if sigma % 2 == 0:
        h = np.zeros(((3 * sigma) + 1, (3 * sigma) + 1))
        x = np.arange(((-3 * sigma)) / 2, (((3 * sigma) + 1) / 2))
        y = np.arange(((-3 * sigma)) / 2, (((3 * sigma) + 1) / 2))
    else:
        h = np.zeros(((3 * sigma), (3 * sigma)))
        x = np.arange(((-3 * sigma) + 1) / 2, (((3 * sigma) + 1) / 2))
        y = np.arange(((-3 * sigma) + 1) / 2, (((3 * sigma) + 1) / 2))
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            sum = sum + (1 / (2 * np.pi * np.power(sigma, 2))) * np.power(np.e, (-((np.power(x[i], 2) + np.power(y[j], 2)) / (2 * np.power(sigma, 2)))))
            h[int(i), int(j)] = (1 / (2 * np.pi * np.power(sigma, 2))) * np.power(np.e, (-((np.power(x[i], 2) + np.power(y[j], 2)) / (2 * np.power(sigma, 2)))))
    h = h / sum
    return h

def myfilter(I,h):
    #Appropriately pad I
    xidim, yidim = np.array(np.shape(I))
    xhdim, yhdim = np.array(np.shape(h))
    xdim = xidim - xhdim
    ydim = yidim - yhdim
    #I_padded = np.pad(I,((int(xhdim / 2), int(xhdim / 2)), (int(yhdim / 2), int(yhdim / 2))), 'constant', constant_values=0)
    I_padded = I
    h_padded = np.pad(h,((int(xdim / 2) + 1, int(xdim - (xdim / 2))), (int(ydim / 2) + 1, int(ydim - (ydim / 2)))), 'constant', constant_values=0)
    # Convolve I with h
    I_filtered = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(h_padded) * np.fft.fft2(I_padded)).real)
    #remove the following comments to show image
    #plt.title('WIth filter h3')
    plt.imshow(I_filtered, interpolation='none', cmap='gray')
    plt.axis('off')
    plt.show()
    return I_filtered


h1=np.array([[-1/9,-1/9,-1/9],[-1/9,2,-1/9],[-1/9,-1/9,-1/9]])
h2=np.array([[-1,3,-1]])
h3=np.array([[-1],[3],[-1]])


### Part 2
I = np.array(open('Iribe.jpg').convert('L'))/255
Sx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
Sy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
def myCanny(I,sigma=1,t_low=.5,t_high=1):
    #Smooth with gaussian kernel
    I_filtered = np.copy(myfilter(I,gausskernel(sigma)))
    I_edge = np.copy(I_filtered)
    #Find img gradients
    dx = np.copy(myfilter(I_filtered,Sx))
    dy = np.copy(myfilter(I_filtered,Sy))
    amp = np.hypot(dy, dx)
    amp = amp / amp.max() * 255.
    angle = np.copy(np.arctan2(dy, dx))
    def mag(x,y):
        #(np.sqrt((np.power(dx, 2)) + (np.power(dy, 2)))[x,y])
        return amp[x,y]
    def theta(x,y):
        degree = angle[x,y]
        if (degree >= -(np.pi/8) and degree < np.pi/8) or (degree >= 7*np.pi/8 or degree < -7*np.pi/8):
            return 0
        elif (degree >= np.pi/8 and degree < 3*np.pi/8) or (degree >= -3*np.pi/8 and degree < -(np.pi/8)):
            return 45
        elif (degree >= 3*np.pi/8 and degree < 5*np.pi/8) or (degree >= -5*np.pi/8 and degree < -3*np.pi/8):
            return 90
        elif (degree >= 5*np.pi/8 and degree < 7*np.pi/8) or (degree >= -7*np.pi/8 and degree < -5*np.pi/8):
            return 135
    #Thin edges
    def thin(x,y):
        x1 = x
        y1 = y
        x2 = x
        y2 = y
        a = theta(x,y)
        if(a == 0):
            y1 = y + 1
            y2 = y - 1
        elif (a == 45):
            x1 = x - 1
            x2 = x + 1
        elif (a == 90):
            y1 = y + 1
            x1 = x - 1
            y2 = y - 1
            x2 = x + 1
        elif (a == 135):
            y1 = y + 1
            x1 = x + 1
            y2 = y - 1
            x2 = x - 1
        else:
            print('you shouldnt see this')
        comp = mag(x,y)
        if(x1 >= 0 and y1 >= 0 and x1 < xdim and y1 < ydim):
            comp = max(comp,mag(x1,y1))
        if(x2 >= 0 and y2 >= 0 and x2 < xdim and y2 < ydim):
            comp = max(comp,mag(x2,y2))
        if(comp <= mag(x,y) and mag(x,y) > t_low):
            I_edge[x,y] = mag(x,y)
        else:
            I_edge[x,y] = 0.
    xdim, ydim = np.shape(I_filtered)
    print('starting thinning')
    for i in range(0,xdim):
        for j in range(0, ydim):
            thin(i,j)
    print('finish thinning')
    #Hystersis thresholding
    print('start hystersis')
    from scipy.ndimage.measurements import label
    labels, num_feat = label(I_edge)
    print('did label')
    for n in range(1, num_feat + 1):
        contain = np.where(labels == n)
        x = contain[0]
        y = contain[1]
        high = False
        for k in range(len(x)):
            if mag(x[k],y[k]) > t_high:
                high = True
        for k in range(len(x)):
            if mag(x[k],y[k]) < t_low:
                I_edge[x[k],y[k]] = 0.
            elif high:
                I_edge[x[k],y[k]] = 1.
            else:
                I_edge[x[k],y[k]] = 0.

    myedges = I_edge
    print('finish hystersis')
    return myedges

sigma = 3
t_low = 10
t_high = 20
print('doing mycanny')
edges=myCanny(I,sigma,t_low,t_high)
print('finished mycanny')
plt.title('sigma = ' + str(sigma) + ' t_low = ' + str(t_low) + ' t_high = ' + str(t_high))
plt.imshow(edges, interpolation='none',cmap='gray')
plt.show()
print('done')