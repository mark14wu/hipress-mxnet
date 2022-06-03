import numpy as np
import mxnet as mx
import time

# def binary(num):
#     import struct
#     return ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))

# def float32_to_int32(num):
#     b = binary(num)
#     if b[0]=='0':
#         return int(b,2)
#     else:
#         return -int(b[1:],2)
# def N_to_M(N):
#     return N*4

# def verify_mem_comp():
#     grad = [0, 0, 1]
#     u = [1, 1, 1]
#     v = [0, 3, 10]
#     ctx = mx.context.gpu(0)

#     grad = mx.nd.array(grad, ctx=ctx, dtype='float32')
#     u = mx.nd.array(u, ctx=ctx, dtype='float32')
#     v = mx.nd.array(v, ctx=ctx, dtype='float32')
#     momentum = 0.5

#     true_u = momentum * u + grad
#     true_v = true_u + v
#     mx.nd.contrib.dgc_mem_comp(
#         grad=grad,
#         u=u,
#         v=v,
#         # data=grad,
#         momentum=momentum
#     )
#     u.wait_to_read()
#     v.wait_to_read()
#     print(u)
#     print(true_u)
#     print(v)
#     print(true_v)
#     assert sum(u - true_u) == 0
#     assert sum(v - true_v) == 0
#     print("test passed!")

def verify_topk():
    u = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10
    # verify_topk_adam(u)
    verify_topk_adam_encode(u)
    # v = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] + 10 * [0]
    # verify_dgc(v, v)

# def verify_dgc_with_residual_clear():
#     g = [1, 1, 0, 0, 5, 5, 5, 0, 0, 0] + 10 * [0]
#     u = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + 10 * [0]
#     v = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] + 10 * [0]
#     verify_dgc(g, u, v)

#     # u = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0] + 10 * [0]
#     # v = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + 10 * [0]
#     # verify_dgc(u, v)

# def get_size_in_header(A):
#     A = mx.nd.cast(A, dtype='int32')
#     return A[0] + 256 * A[1] + 65536 * A[2] + 256 * 65536 * A[3]

def verify_topk_adam_encode(grad):
    ctx = mx.gpu()
    encode = mx.nd.contrib.topk_adam_encode
    N = len(grad)
    to_compress = mx.nd.array(grad, ctx=ctx, dtype='float32')
    compressed = mx.nd.zeros(shape=4 * N, ctx=ctx, dtype='uint8')
    iterations = 100000
    for i in range(iterations):
        encode(
            residual=to_compress,
            s_percent=0.3,
            sample_rate=1.0,
            out=compressed
        )
        compressed.wait_to_read()

def verify_topk_adam(grad):
    ctx = mx.context.gpu(0)
    encode = mx.nd.contrib.topk_adam_encode
    server_encode = mx.nd.contrib.topk_adam_server_encode
    decode = mx.nd.contrib.topk_adam_decode
    N = len(grad)

    to_compress = mx.nd.array(grad, ctx=ctx, dtype='float32')
    compressed = mx.nd.zeros(shape=4 * N, ctx=ctx, dtype='uint8')
    decompressed = mx.nd.zeros(shape=N, ctx=ctx, dtype='float32')

    print("before topk_adam:")
    print("adam:", to_compress)

    print("topk_adam...")
    encode(
        residual=to_compress,
        s_percent=0.3,
        sample_rate=1.0,
        out=compressed
    )
    compressed.wait_to_read()

    print('decode...')
    decode(
        data=compressed,
        out=decompressed
    )
    decompressed.wait_to_read()
    print("validating decompressed...")
    print("grad", to_compress)
    print("decompressed", decompressed)

# def verify_dgc(grad, u, v):
#     ctx = mx.context.gpu(0)
#     dgc = mx.nd.contrib.dgc_new
#     dgcr = mx.nd.contrib.dgc_new_r
#     N = len(u)

#     to_compress = mx.nd.array(grad, ctx=ctx, dtype='float32')
#     input_u = mx.nd.array(u, ctx=ctx, dtype='float32')
#     input_v = mx.nd.array(v, ctx=ctx, dtype='float32')
#     compressed = mx.nd.zeros(shape=4 * N, ctx=ctx, dtype='uint8')
#     decompressed = mx.nd.zeros(shape=N, ctx=ctx, dtype='float32')

#     print("before dgc:")
#     print("grad", to_compress)
#     print("u", input_u)
#     print("v", input_v)

#     print("dgc...")
#     dgc(
#         grad=to_compress,
#         u=input_u,
#         v=input_v,
#         s_percent=0.151,
#         sample_rate=1.0,
#         momentum=0.1,
#         out=compressed
#     )
#     # compressed.wait_to_read()

#     print("M:", compressed[:4])
#     print("size:", get_size_in_header(compressed[:4]))
#     print('dgcr...')
#     dgcr(
#         data=compressed,
#         out=decompressed
#     )
#     # decompressed.wait_to_read()
#     print("validating decompressed...")
#     b = decompressed.asnumpy()
#     print("grad", to_compress)
#     print("u", input_u)
#     print("v", input_v)
#     # print("b", decompressed)
#     print("b", b)
#     print("dgc validating pass!")

# def verify_server_encode(N=2**6, xpu='gpu'):
#     assert(xpu in ['cpu','gpu','omp'])
#     if xpu == 'gpu':
#         ctx = mx.context.gpu(0)
#     else:
#         ctx = mx.context.cpu(0)

#     dgc_server = mx.nd.contrib.dgc_new_server
#     dgcr = mx.nd.contrib.dgc_new_r

#     a = np.random.randint(0, 100, N)
#     a[a > 10] = 0
#     to_compress = mx.nd.array(a, ctx=ctx, dtype='float32')
#     compressed = mx.nd.zeros(shape=4 * (N * 2 + 1), ctx=ctx, dtype='uint8')
#     decompressed = mx.nd.zeros(shape=N, ctx=ctx, dtype='float32')

#     print("dgc_server...")
#     dgc_server(
#         data=to_compress,
#         out=compressed
#     )
#     compressed.wait_to_read()

#     print("dgcr...")
#     dgcr(
#         data=compressed,
#         out=decompressed
#     )
#     decompressed.wait_to_read()

#     print('validating decompressed...')
#     b = decompressed.asnumpy()
#     for i in range(N):
#         if to_compress[i] != b[i]:
#             print("i={}\ta[i]={}\tb[i]={}".format(i,to_compress[i],b[i]))
#             assert(0)
#     print("dgc_server validating pass!")

# def verify_xpu(N,s_percent,sample_rate,xpu):
#     assert(xpu in ['cpu','gpu','omp'])
#     if xpu == 'gpu':
#         ctx = mx.context.gpu(0)
#     else:
#         ctx = mx.context.cpu(0)
#     dgc = mx.nd.contrib.dgc_new
#     dgcr = mx.nd.contrib.dgc_new_r

#     import random
#     # a = [random.randint(-N//2,N//2) for i in range(N)]
#     a = [random.normalvariate(0,0.5) for i in range(N)]
#     to_compress = mx.nd.array(a,ctx=ctx,dtype='float32')
#     import math
#     M = N_to_M(N)
#     compressed = mx.nd.zeros(shape=(M),ctx=ctx,dtype='uint8')
#     decompressed = mx.nd.zeros(shape=(N),ctx=ctx,dtype='float32')
    
#     print("dgc...")
#     dgc(
#         data=to_compress,
#         s_percent = s_percent,
#         sample_rate = sample_rate,
#         out = compressed
#     )
#     compressed.wait_to_read()

#     print('dgcr...')
#     dgcr(
#         data=compressed,
#         out = decompressed
#     )
#     decompressed.wait_to_read()
#     print("validating decompressed...")
#     b = decompressed.asnumpy()
#     cnt = 0
#     for i in range(N):
#         if b[i] > 1e-6 or b[i] < -1e-6:
#             cnt += 1
#             error = abs((a[i]-b[i])/b[i])
#             if error<1e-6:
#                 print("i={}\ta[i]={}\tb[i]={}\tv={}".format(i,a[i],b[i],))
#                 assert(0)
#     print("dgcr_thrust validating pass!")

# def num_float_to_size(n):
#     def deal_float(v):
#         if int(v)==v:
#             return int(v)
#         else:
#             return v
#     size_B = n*4
#     if size_B < 1024:
#         return '{}B'.format(deal_float(size_B))
#     size_KB = size_B/1024
#     if size_KB < 1024:
#         return '{}KB'.format(deal_float(size_KB))
#     size_MB = size_KB/1024
#     if size_MB < 1024:
#         return '{}MB'.format(deal_float(size_MB))
#     size_GB = size_MB/1024
#     return '{}GB'.format(deal_float(size_GB))
    
# def benchmark_xpu(N:int, s_percent:float, sample_rate:float, xpu:str, container:dict):
#     assert(xpu in ['cpu','gpu'])
#     if xpu == 'gpu': 
#         ctx = mx.context.gpu(0)
#     else:
#         ctx = mx.context.cpu(0)
#     dgc = mx.nd.contrib.dgc_new
#     dgcr = mx.nd.contrib.dgc_new_r

#     import math
#     M = N_to_M(N)
#     to_compress = container[xpu]['float32'][:N]
#     temp = container['gpu']['float32'][:N]
#     compressed = container[xpu]['uint8'][:M]
#     decompressed = container[xpu]['float32'][N:2*N]
#     import random
#     compression_time = []
#     decompression_time = []
#     times = 101
#     for i in range(times):
#         # print('i={}'.format(i))
#         # mx.nd.contrib.float_random(param1=0,param2=0.5,type=1,out=temp)
#         mx.nd.contrib.float_random(param1=-2,param2=2,type=0,out=temp)
#         temp.copyto(to_compress)
#         to_compress.wait_to_read()
#         import time
#         # print("dgc...")
#         t1 = time.time()
#         dgc(
#             data=to_compress,
#             s_percent = s_percent,
#             sample_rate = sample_rate,
#             out = compressed
#         )
#         compressed.wait_to_read()
#         t2 = time.time()
#         # print("dgcr...")
#         t3 = time.time()
#         dgcr(
#             data = compressed,
#             out = decompressed
#         )
#         decompressed.wait_to_read()
#         t4 = time.time()
#         if i > 0:
#             compression_time.append(t2-t1)
#             decompression_time.append(t4-t3)
#     compression_time.sort()
#     decompression_time.sort()
#     head = int(len(compression_time)*0.25)
#     tail = int(len(compression_time)*0.75)
#     compression_time = compression_time[head:tail]
#     decompression_time = decompression_time[head:tail]
#     return sum(compression_time)/len(compression_time), sum(decompression_time)/len(decompression_time)

# def benchmark_all():
#     import math
#     container = dict()
#     max_N_lg2 = 27
#     max_N = 2**max_N_lg2
#     pre_allocated_size = 2*max_N
#     max_M = N_to_M(max_N)
#     c_cpu = dict()
#     c_cpu['float32'] = mx.nd.zeros(shape=pre_allocated_size, ctx=mx.context.cpu(0),dtype='float32')
#     c_cpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.cpu(0), dtype='uint8')
#     c_gpu = dict()
#     c_gpu['float32'] = mx.nd.zeros(shape=pre_allocated_size, ctx=mx.context.gpu(0), dtype='float32')
#     c_gpu['uint8'] = mx.nd.zeros(shape=max_M, ctx=mx.context.gpu(0), dtype='uint8')
#     container['cpu'] = c_cpu
#     container['gpu'] = c_gpu
#     output_csv = "dgc_new.csv"
#     with open(output_csv, 'w') as f:
#         f.write('N,size,xpu,s_percent,sample_rate,ct(us),dt(us)\n')
#     for N_lg2 in range(max_N_lg2-2,max_N_lg2+1):
#         for sample_rate in [0.001]:
#             for s_percent in [0.001]:
#                 for xpu in ['gpu']:
#                 # for xpu in ['cpu','gpu']:
#                     N = 2**N_lg2
#                     result = benchmark_xpu(N,s_percent,sample_rate,xpu,container)
#                     line = "{N},{size},{xpu},{s_percent},{sample_rate},{ct},{dt}".format(
#                         N=N,
#                         size = num_float_to_size(N),
#                         xpu = xpu,
#                         s_percent = s_percent,
#                         sample_rate = sample_rate,
#                         ct = result[0],
#                         dt = result[1]
#                     )
#                     print(line)
#                     with open(output_csv,'a') as f:
#                         f.write(line+'\n')
    
# def verify_all():
#     for i in range(6,20):
#         print("verifying {}".format(2**i))
#         verify_xpu(2**i, 0.001, 0.001, 'gpu')
#         # verify_xpu(2**i, 0.001, 0.001, 'cpu')

if __name__ == '__main__':
    verify_topk()

