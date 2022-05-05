import numpy as np
import mxnet as mx
import time

output_csv = r"gpu_nonrandom_3.csv"

def check(a,b,bitwidth=2):
    data_per_byte = 8//bitwidth
    min_f = min(a)
    max_f = max(a)
    gap = (max_f - min_f) / (data_per_byte-1)
    for i in range(len(a)):
        t = (a[i] - min_f) / gap
        v = round(t) * gap + min_f
        if abs(v-b[i]) > 0.01:
            print(i,a[i],b[i],t,v)
            print("failed")
            #assert(0)
    print("check pass!")
    return True




def test_time(input_size, bitwidth, random):
    print("testing: input_size={},bitwidth={},random={}".format(input_size,bitwidth,random))
    output_size = 10 + (input_size+4-1)//4
    tail = input_size % 4;
    tail = 0 if tail == 0 else 4 - tail

    a = np.random.randn(input_size)
    b = mx.nd.array(a,ctx=mx.context.gpu(0),dtype='float32')
    c = mx.nd.array(a[:output_size],ctx=mx.context.gpu(0),dtype='uint8')

    total_cnt = 10
    total_time = 0
    total_time_reverse = 0
    mx.nd.contrib.terngrad(data=b,bitwidth=bitwidth,random=random,out=c)
    c.wait_to_read()
    t1 = time.time()
    for i in range(total_cnt):
        mx.nd.contrib.terngrad(data=b,bitwidth=bitwidth,random=random,out=c)
        c.wait_to_read()
    t2 = time.time()
    total_time = t2 -t1

    mx.nd.contrib.terngradr(data=c,bitwidth=bitwidth,tail=tail,out=b)
    b.wait_to_read()

    t1 = time.time()
    for i in range(total_cnt):
        mx.nd.contrib.terngradr(data=c,bitwidth=bitwidth,tail=tail,out=b)
        b.wait_to_read()
    t2 = time.time()
    total_time_reverse = t2 - t1

    average_time = total_time / total_cnt
    average_time_reverse = total_time_reverse / total_cnt
    check(a,b.asnumpy())
    print("average_time={}".format(average_time))
    print("average_time_reverse={}".format(average_time_reverse))
    with open(output_csv,'a') as f:
        line = ["terngrad",input_size,bitwidth,random,total_time,total_cnt,average_time]
        line = [str(v) for v in line]
        line = ','.join(line)
        f.write(line+'\n')
        line = ["terngrad_reverse",input_size,bitwidth,random,total_time_reverse,total_cnt,average_time_reverse]
        line = [str(v) for v in line]
        line = ','.join(line)
        f.write(line+'\n')

if __name__ == '__main__':
    with open(output_csv,'w') as f:
        attr = ["operator","size","bitwidth","random","total_time","total_cnt","average_time"]
        line = ','.join(attr)
        f.write(line+'\n')
    
    max_size = 2**27
    input_size = 32
    while input_size <= max_size:
        #for bitwidth in [1,2,4,8]:
        for bitwidth in [2]:
            for random in [0]:
                test_time(input_size,bitwidth,random)
                # test_time(131072,bitwidth,random)
        input_size *= 2
        #time.sleep(10)



