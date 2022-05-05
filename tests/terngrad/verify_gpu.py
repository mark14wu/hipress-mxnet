import numpy as np
import mxnet as mx
import time

def check(a,b,bitwidth=2):
    data_per_byte = 8//bitwidth
    min_f = min(a)
    max_f = max(a)
    gap = (max_f - min_f) / (data_per_byte-1)
    for i in range(len(a)):
        t = (a[i] - min_f) / gap
        v = round(t) * gap + min_f
        if b[i]!=0 and abs((v-b[i])/b[i])>0.001:
            print(i,a[i],b[i],t,v)
            print("failed")
            #assert(0)
    print("check pass!")
    return True


def benchmark_xpu(N,random,xpu):
    assert(xpu in ['cpu','gpu','omp'])
    if xpu == 'gpu':
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu()
    if xpu == 'omp':
        terngrad = mx.nd.contrib.terngrad_omp
        terngradr = mx.nd.contrib.terngradr_omp
    else:
        terngrad = mx.nd.contrib.terngrad
        terngradr = mx.nd.contrib.terngradr
    # print("preparing a...\tN={}".format(N))
    a = [i+1 for i in range(N)]
    a = np.array(a)
    # print("a prepared!")
    # print("preparing b and c...")
    b = mx.nd.array(a,ctx=ctx,dtype='float32')
    c = mx.nd.array(a[:10+(N+3)//4],ctx=ctx,dtype='uint8')
    # print("b and c prepared!")
    compression_time = []
    decompression_time = []
    times = 11
    for i in range(times):
        t1 = time.time()
        terngrad(data=b, out=c)
        c.wait_to_read()
        t2 = time.time()

        bitwidth = c[0].asnumpy()[0]
        tail = c[1].asnumpy()[0]

        t3 = time.time()
        terngradr(data=c,bitwidth=bitwidth,tail=tail,out=b)
        b.wait_to_read()
        t4 = time.time()
        if i > 0:
            compression_time.append(t2-t1)
            decompression_time.append(t4-t3)
    return sum(compression_time)/len(compression_time),sum(decompression_time)/len(decompression_time)

def num_float_to_size(n):
    def deal_float(v):
        if int(v)==v:
            return int(v)
        else:
            return v
    size_B = n*4
    if size_B < 1024:
        return '{}B'.format(deal_float(size_B))
    size_KB = size_B/1024
    if size_KB < 1024:
        return '{}KB'.format(deal_float(size_KB))
    size_MB = size_KB/1024
    if size_MB < 1024:
        return '{}MB'.format(deal_float(size_MB))
    size_GB = size_MB/1024
    return '{}GB'.format(deal_float(size_GB))
    


if __name__ == '__main__':

    output_csv = "terngrad.csv"
    with open(output_csv,'w') as f:
        f.write("")
    
    for N_lg2 in range(5,28):
        for random in [0]:
            for xpu in ['gpu','cpu','omp']:
                N = 2**N_lg2
                result = benchmark_xpu(N,random,xpu)
                print("N:{}\trandom:{}\txpu:{}\tct:{}\tdt:{}".format(
                    N,random,xpu,result[0],result[1]
                ))
                with open(output_csv,'a') as f:
                    f.write("{N},{size},{random},{xpu},{ct},{dt}\n".format(
                        N=N,
                        size=num_float_to_size(N),
                        random=random,
                        xpu=xpu,
                        ct=result[0],
                        dt=result[1]
                    ))
    exit(0)
    #a = [(i) % 300000 for i in range(2**27)]
    N = 2**11-1
    N = 1024
    N = 2050
    N = 34
    N = 2050
    N = 2**27
    N = 131072
    N = 2**5
    N = 2**27
    #a = [N-i for i in range(N)]

    xpu = 'omp'
    assert(xpu in ['cpu','gpu','omp'])
    if xpu == 'gpu':
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu(0)
    
    if xpu == 'omp':
        terngrad = mx.nd.contrib.terngrad_omp
        terngradr = mx.nd.contrib.terngradr_omp
    else:
        terngrad = mx.nd.contrib.terngrad
        terngradr = mx.nd.contrib.terngradr

    print("preparing a... N = {}".format(N))
    a = [i+1 for i in range(N)]
    a = np.array(a)
    print("a prepared!")
    # b = mx.nd.array(a,ctx=mx.context.gpu(0),dtype="float32")
    b = mx.nd.array(a,ctx=ctx,dtype="float32")
    # c = mx.nd.array(a[:10+(N+3)//4],ctx=mx.context.gpu(0),dtype="uint8")
    c = mx.nd.array(a[:10+(N+3)//4],ctx=ctx,dtype="uint8")
    print("b prepared!")
    print("b.ctx:",b.context)
    print("original data before quantize:", b.asnumpy())
    terngrad(data=b,out=c)
    c.wait_to_read()

    #print("d[:20]",d[:20])

    times = 200
    t1 = time.time()
    for i in range(times):
        terngrad(data=b,random=0,out=c)
        c.wait_to_read()
    t2 = time.time()
    print("quantized data:",c.asnumpy())
    #print("original data after quantize:",b.asnumpy())
    print("quantization cost time (s):",(t2-t1)/times)


    d = c.asnumpy()
    bitwidth = d[0]
    tail = d[1]

    terngradr(data=c,bitwidth = bitwidth, tail = tail, out=b)
    b.wait_to_read()
    t1 = time.time()
    for i in range(times):
        terngradr(data=c,bitwidth = bitwidth, tail = tail, out=b)
        b.wait_to_read()
    t2 = time.time()

    d = b.asnumpy()
    print("dequantized data:",d)
    print("dequantization cost time (s):",(t2-t1)/times)
    print("checking correctness...")
    check(a,d)
    #print("d[:20]",d[:20])
    exit()







