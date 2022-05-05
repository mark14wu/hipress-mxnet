import numpy as np
import mxnet as mx
import time
import math

def benchmark_xpu(N,bitwidth,alpha,beta,xpu,container):
    assert(bitwidth in [2,4,8])
    assert(xpu in ['cpu','gpu'])
    if xpu == 'gpu':
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu(0)
    ecq_sgd = mx.nd.contrib.ecq_sgd
    ecq_sgd_r = mx.nd.contrib.ecq_sgd_r
    data_per_byte = 8 / bitwidth
    M = 10 + math.ceil(N/data_per_byte) + N*4
    to_compress = container[xpu]['float32'][:N]
    compressed = container[xpu]['uint8'][:M]
    decompressed = container[xpu]['float32'][N:2*N]
    residual = container[xpu]['float32'][2*N:3*N]
    temp = container['gpu']['float32'][:N]
    compression_time = []
    decompression_time = []
    times = 101
    for i in range(times):
        mx.nd.contrib.float_random(param1=0,param2=0.5,type=1,out=temp)
        temp.copyto(to_compress)
        to_compress.wait_to_read()
        import time
        t1 = time.time()
        ecq_sgd(
            data = to_compress,
            residual = residual,
            bitwidth = bitwidth,
            alpha = alpha,
            beta = beta,
            out = compressed
        )
        compressed.wait_to_read()
        t2 = time.time()
        t3 = time.time()
        ecq_sgd_r(
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
    container = dict()
    max_N_lg2 = 27
    max_N = 2**max_N_lg2
    pre_allocated_size = 3 * max_N
    min_data_per_byte = 1
    max_M = 10 + math.ceil(max_N / min_data_per_byte) + max_N*4
    c_cpu = dict()
    c_cpu['float32'] = mx.nd.zeros(shape=pre_allocated_size,ctx=mx.context.cpu(0),dtype='float32')
    c_cpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.cpu(0), dtype='uint8')
    c_gpu = dict()
    c_gpu['float32'] = mx.nd.zeros(shape=pre_allocated_size, ctx=mx.context.gpu(0), dtype='float32')
    c_gpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.gpu(0), dtype='uint8')
    container['cpu'] = c_cpu
    container['gpu'] = c_gpu
    output_csv = "ecq_sgc.csv"
    with open(output_csv, 'w') as f:
        f.write("N,bitwidth,alpha,beta,xpu,ct,dt\n")
    for N_lg2 in range(max_N_lg2,max_N_lg2+1):
        # for bitwidth in [2,4,8]:
        for bitwidth in [2]:
            for alpha in [0.2]:
                for beta in [0.9]:
                    for xpu in ['gpu']:
                        N = 2**N_lg2
                        result = benchmark_xpu(N,bitwidth,alpha,beta,xpu,container)
                        line = '{N},{bitwidth},{alpha},{beta},{xpu},{ct},{dt}'.format(
                            N = N,
                            bitwidth = bitwidth,
                            alpha = alpha,
                            beta = beta,
                            xpu = xpu,
                            ct = result[0],
                            dt = result[1]
                        )
                        with open(output_csv,'a') as f:
                            f.write(line+"\n")
                        print(line)

def verify_xpu(N,bitwidth,alpha,beta,xpu):
    assert(bitwidth in [2,4,8])
    assert(xpu in ['cpu','gpu'])
    if xpu == 'gpu':
        ctx = mx.context.gpu(0)
    else:
        ctx = mx.context.cpu(0)
    ecq_sgd = mx.nd.contrib.ecq_sgd
    ecq_sgd_r = mx.nd.contrib.ecq_sgd_r

    import random
    a = [random.uniform(-2,2) for i in range(N)]
    to_compress = mx.nd.array(a,ctx=ctx,dtype='float32')
    residual = mx.nd.zeros(shape=N,ctx=ctx,dtype='float32')
    data_per_byte = 8 / bitwidth
    M = 10 + math.ceil(N/data_per_byte)
    compressed = mx.nd.zeros(shape=M,ctx=ctx,dtype='uint8')
    decompressed = mx.nd.zeros(shape=N,ctx=ctx,dtype='float32')
    print("ecq_sgd...")
    print("verify_xpu:\tbitwidth={}".format(bitwidth))
    ecq_sgd(
        data = to_compress,
        residual = residual,
        bitwidth = bitwidth,
        alpha = alpha,
        beta = beta,
        out = compressed
    )
    compressed.wait_to_read()
    print("ecq_sgd_r...")
    ecq_sgd_r(
        data = compressed,
        out = decompressed
    )
    decompressed.wait_to_read()
    print("to_check...")
    G = to_compress.asnumpy()
    H = residual.asnumpy()
    G_ = decompressed.asnumpy()
    len_print = 16
    print(G[:len_print])
    print(G_[:len_print])
    print(H[:len_print])



def verify_all():
    verify_xpu(2**10, 2, 0.8, 0.8, 'cpu')
    verify_xpu(2**10, 2, 0.8, 0.8, 'gpu')


if __name__ == '__main__':
    # verify_all()
    benchmark_all()