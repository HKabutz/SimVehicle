import numpy as np

alpha = 0

def rotate(px,py,angle,ox,oy):
    pnx = np.cos(angle)*(px-ox)-np.sin(angle)*(py-oy)+ox
    pny = np.sin(angle)*(px-ox)+np.cos(angle)*(py-oy)+oy
    return pnx, pny

xe, ye = rotate(1,1,-1,0,0)
print(xe,"   ",ye)

w = 6
q = 5
w = max(-q, min(w, q))
print(w)