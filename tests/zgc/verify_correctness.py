import numpy as np
import mxnet as mx
import time
import random

def check(original,residual,decompress,threshold):
    assert(len(original) == len(residual) == len(decompress))
    for i in range(len(original)):
        v = original[i]
        if (v >= threshold and (residual[i] != v-threshold or decompress[i] != threshold)) or \
            (v<=-threshold and (residual[i] != v+threshold or decompress[i] !=-threshold)) or \
            (-threshold < v < threshold and (residual[i] != v or decompress[i] != 0)):
            print("check failed! i={i},original[{i}]={original_value}\t"\
                        "residual[{i}]={residual_value}\t"\
                        "decompress[{i}]={decompress_value}\t"\
                        "threshold={threshold}".format(
                            i=i,
                            original_value = original[i],
                            residual_value = residual[i],
                            decompress_value = decompress[i],
                            threshold=threshold
                        ))
            assert(0)
    print("Check pass!")
    return True
        
                


def verify_xpu(N,threshold,xpu):
    assert(xpu in ['cpu','gpu','omp'])
    if xpu == 'gpu':
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu(0)
    if xpu == 'omp':
        zgc = mx.nd.contrib.zgc_omp
        zgcr = mx.nd.contrib.zgcr_omp
    else:
        zgc = mx.nd.contrib.zgc
        zgcr = mx.nd.contrib.zgcr

    a = [N/2 -i for i in range(N)]
    a = np.array(a,dtype=np.float)
    b = np.zeros(shape=(N),dtype=np.float)
    to_compress = mx.nd.array(a, ctx=ctx, dtype = 'float32')
    after_decompress = mx.nd.array(a, ctx=ctx, dtype = 'float32')
    residual = mx.nd.array(b, ctx=ctx, dtype='float32')
    out = mx.nd.zeros(shape=(N), ctx=ctx, dtype='uint8')
    zgc(
        to_compress=to_compress,
        residual=residual,
        threshold=threshold,
        out = out
    )
    c = out.asnumpy()
    zgcr(
        to_decompress = out,
        threshold = threshold,
        out = after_decompress
    )
    import math
    output_size = math.ceil(N/16)
    if 0:
        print("original data:",a)
        print("c[:{output_size}](quantization data):".format(output_size=output_size),c[:output_size])
        print("quantization data as bin: ", end='')
        for i in range(output_size):
            t = bin(c[i])
            t = t[t.find('b')+1:]
            t = (8-len(t))*'0' + t
            print(t,end=' ')
        print()
        print("residual data:", residual.asnumpy())
        print("after_decompress data:",after_decompress.asnumpy())
    print("checking... N={}\tthreshold={}\txpu={}".format(N,threshold,xpu))
    check(a,residual.asnumpy(),after_decompress.asnumpy(),threshold)


    
def N_to_M(N):
    import math
    return math.ceil(N/4)

def benchmark_xpu(N,threshold,xpu,container):
    assert(xpu in ['cpu','gpu','omp'])
    if xpu == 'gpu':
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu(0)
    if xpu == 'omp':
        zgc = mx.nd.contrib.zgc_omp
        zgcr = mx.nd.contrib.zgcr_omp
    else:
        zgc = mx.nd.contrib.zgc
        zgcr = mx.nd.contrib.zgcr

    import math

    M = N_to_M(N)
    to_compress = container[xpu]['float32'][:N]
    residual = container[xpu]['float32'][N:2*N]
    decompressed = to_compress
    M = N_to_M(N)
    compressed = container[xpu]['uint8'][:M]
    temp = container['gpu']['float32'][:N]

    compression_time = []
    decompression_time = []
    times = 101 if xpu == 'gpu' else 11
    for i in range(times):
        mx.nd.contrib.float_random(param1=-threshold*1.5,param2=threshold*1.5,type=0,out=temp)
        temp.copyto(to_compress)
        to_compress.wait_to_read()
        import time
        t1 = time.time()
        zgc(
            to_compress = to_compress,
            residual = residual,
            threshold = threshold,
            out = compressed
        )
        compressed.wait_to_read()
        t2 = time.time()
        t3 = time.time()
        zgcr(
            to_decompress = compressed,
            threshold = threshold,
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
    
def verify_all():
    for N in [2**6, 2**6+2, 2**20]:
        threshold = N/4
        for xpu in ['cpu','omp','gpu']:
            verify_xpu(N,threshold,xpu)

def benchmark_all():
    container = dict()
    max_N_lg2 = 27
    max_N = 2**max_N_lg2
    pre_allocated_size = max_N*2
    max_M = N_to_M(max_N)
    c_cpu = dict()
    c_cpu['float32'] = mx.nd.zeros(shape=pre_allocated_size, ctx=mx.context.cpu(0),dtype='float32')
    c_cpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.cpu(0),dtype='uint8')
    c_gpu = dict()
    c_gpu['float32'] = mx.nd.zeros(shape=pre_allocated_size, ctx=mx.context.gpu(0), dtype='float32')
    c_gpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.gpu(0),dtype='uint8')
    container['cpu'] = c_cpu
    container['gpu'] = c_gpu
    container['omp'] = c_cpu

    output_csv = "zgc_gpu_lite.csv"
    with open(output_csv,'w') as f:
        f.write("N,size,xpu,compression_time,decompresstion_time\n")

    for N_lg2 in range(6, max_N_lg2+1):
        N = 2**N_lg2
        threshold = 1
        # for xpu in ['cpu','omp','gpu']:
        # for xpu in ['gpu']:
        for xpu in ['cpu','gpu']:
            result = benchmark_xpu(N,threshold,xpu,container)
            line = "{N},{size},{xpu},{ct},{dt}".format(
                    N=N, 
                    size = num_float_to_size(N),
                    xpu = xpu,
                    ct = result[0],
                    dt = result[1]
                )
            print(line)
            with open(output_csv,'a') as f:
                f.write(line+'\n')


if __name__ == '__main__':
    # verify_all()
    benchmark_all()



    



