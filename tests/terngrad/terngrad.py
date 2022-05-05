import numpy as np
import mxnet as mx
import time

output_csv = r"cpu_single_thread.csv"


def test_time(input_size, bitwidth, random):
    print("testing: input_size={},bitwidth={},random={}".format(input_size,bitwidth,random))
    total_time = 0
    total_cnt = 0
    total_time_reverse = 0
    while total_cnt < 10:
        #time.sleep(1)
        total_cnt+=1
        input_data0 = np.random.randn(input_size)
        input_data = mx.nd.array(input_data0,ctx=mx.context.cpu_pinned())
        output_size = 10 + (input_size+4-1)//4
        output = mx.nd.zeros(shape=(output_size),dtype='uint8',ctx=mx.context.cpu_pinned())
        t1 = time.time()
        #output = mx.nd.contrib.terngrad(data=input_data,bitwidth=bitwidth,random=random)
        mx.nd.contrib.terngrad(data=input_data,bitwidth=bitwidth,random=random,out=output)
        output.wait_to_read()
        t2 = time.time()
        input_reverse = mx.nd.contrib.terngradr(data=output,bitwidth=bitwidth)
        t3 = time.time()
        total_time = total_time + t2 - t1
        total_time_reverse = total_time_reverse + t3 - t2
    average_time = total_time / total_cnt
    average_time_reverse = total_time_reverse / total_cnt
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
    
    max_size = 134217728
    input_size = 32
    while input_size <= max_size:
        #for bitwidth in [1,2,4,8]:
        for bitwidth in [2]:
            for random in [0,1]:
                test_time(input_size,bitwidth,random)
        input_size *= 2



