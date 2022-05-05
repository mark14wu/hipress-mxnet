import numpy as np
import mxnet as mx
import time

def binary(num):
    import struct
    return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

def float32_to_int32(num):
    b = binary(num)
    if b[0]=='0':
        return int(b,2)
    else:
        return -int(b[1:],2)
def N_to_M(N):
    return N*4

def verify_xpu(N,s_percent,sample_rate,xpu):
    assert(xpu in ['cpu','gpu','omp'])
    if xpu == 'gpu':
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu(0)
    dgc = mx.nd.contrib.dgc_new
    dgcr = mx.nd.contrib.dgc_new_r

    import random
    # a = [random.randint(-N//2,N//2) for i in range(N)]
    a = [random.normalvariate(0,0.5) for i in range(N)]
    to_compress = mx.nd.array(a,ctx=ctx,dtype='float32')
    import math
    M = N_to_M(N)
    compressed = mx.nd.zeros(shape=(M),ctx=ctx,dtype='uint8')
    decompressed = mx.nd.zeros(shape=(N),ctx=ctx,dtype='float32')
    
    print("dgc...")
    dgc(
        data=to_compress,
        s_percent = s_percent,
        sample_rate = sample_rate,
        out = compressed
    )
    compressed.wait_to_read()

    print('dgcr...')
    dgcr(
        data=compressed,
        out = decompressed
    )
    decompressed.wait_to_read()
    print("validating decompressed...")
    b = decompressed.asnumpy()
    cnt = 0
    for i in range(N):
        if b[i] > 1e-6 or b[i] < -1e-6:
            cnt += 1
            error = abs((a[i]-b[i])/b[i])
            if error<1e-6:
                print("i={}\ta[i]={}\tb[i]={}\tv={}".format(i,a[i],b[i],))
                assert(0)
    print("dgcr_thrust validating pass!")


    # print('c[:32]',c[:32])

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
    
def benchmark_xpu(N:int, s_percent:float, sample_rate:float, xpu:str, container:dict):
    assert(xpu in ['cpu','gpu'])
    if xpu == 'gpu': 
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu(0)
    dgc = mx.nd.contrib.dgc_new
    dgcr = mx.nd.contrib.dgc_new_r

    import math
    M = N_to_M(N)
    to_compress = container[xpu]['float32'][:N]
    temp = container['gpu']['float32'][:N]
    compressed = container[xpu]['uint8'][:M]
    decompressed = container[xpu]['float32'][N:2*N]
    import random
    compression_time = []
    decompression_time = []
    times = 101
    for i in range(times):
        # print('i={}'.format(i))
        # mx.nd.contrib.float_random(param1=0,param2=0.5,type=1,out=temp)
        mx.nd.contrib.float_random(param1=-2,param2=2,type=0,out=temp)
        temp.copyto(to_compress)
        to_compress.wait_to_read()
        import time
        # print("dgc...")
        t1 = time.time()
        dgc(
            data=to_compress,
            s_percent = s_percent,
            sample_rate = sample_rate,
            out = compressed
        )
        compressed.wait_to_read()
        t2 = time.time()
        # print("dgcr...")
        t3 = time.time()
        dgcr(
            data = compressed,
            out = decompressed
        )
        decompressed.wait_to_read()
        t4 = time.time()
        if i > 0:
            compression_time.append(t2-t1)
            decompression_time.append(t4-t3)
    compression_time.sort()
    decompression_time.sort()
    head = int(len(compression_time)*0.25)
    tail = int(len(compression_time)*0.75)
    compression_time = compression_time[head:tail]
    decompression_time = decompression_time[head:tail]
    return sum(compression_time)/len(compression_time), sum(decompression_time)/len(decompression_time)


def benchmark_all():
    import math
    container = dict()
    max_N_lg2 = 27
    max_N = 2**max_N_lg2
    pre_allocated_size = 2*max_N
    max_M = N_to_M(max_N)
    c_cpu = dict()
    c_cpu['float32'] = mx.nd.zeros(shape=pre_allocated_size, ctx=mx.context.cpu(0),dtype='float32')
    c_cpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.cpu(0), dtype='uint8')
    c_gpu = dict()
    c_gpu['float32'] = mx.nd.zeros(shape=pre_allocated_size, ctx=mx.context.gpu(0), dtype='float32')
    c_gpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.gpu(0), dtype='uint8')
    container['cpu'] = c_cpu
    container['gpu'] = c_gpu
    output_csv = "dgc_new.csv"
    with open(output_csv, 'w') as f:
        f.write('N,size,xpu,s_percent,sample_rate,ct(us),dt(us)\n')
    for N_lg2 in range(max_N_lg2-2,max_N_lg2+1):
        for sample_rate in [0.001]:
            for s_percent in [0.001]:
                for xpu in ['gpu']:
                # for xpu in ['cpu','gpu']:
                    N = 2**N_lg2
                    result = benchmark_xpu(N,s_percent,sample_rate,xpu,container)
                    line = "{N},{size},{xpu},{s_percent},{sample_rate},{ct},{dt}".format(
                        N=N,
                        size = num_float_to_size(N),
                        xpu = xpu,
                        s_percent = s_percent,
                        sample_rate = sample_rate,
                        ct = result[0],
                        dt = result[1]
                    )
                    print(line)
                    with open(output_csv,'a') as f:
                        f.write(line+'\n')




    

def verify_all():
    for i in range(6,20):
        print("verifying {}".format(2**i))
        verify_xpu(2**i, 0.001, 0.001, 'gpu')
        # verify_xpu(2**i, 0.001, 0.001, 'cpu')
if __name__ == '__main__':
    # verify_all()
    benchmark_all()

