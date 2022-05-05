import numpy as np
import mxnet as mx
import time

if __name__ == '__main__':
    a = [i+1 for i in range(34)]
    a = np.array(a)
    #a = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,12,12])
    print("origin data:",a)
    b = mx.nd.array(a)
    e = mx.nd.zeros(19,dtype='uint8')
    print("before quantized e:",e)
    c = mx.nd.contrib.terngrad(data=b,bitwidth=2,random=0)
    print("quantized data without out provided:",c)
    c = mx.nd.contrib.terngrad(data=b,bitwidth=2,random=0,out=e)
    print("quantized data with out provided:",c)
    print("after quantized e:",e)
    d = mx.nd.contrib.terngradr(data=c,bitwidth=2,tail=2)
    print("dequantized data:",d)



