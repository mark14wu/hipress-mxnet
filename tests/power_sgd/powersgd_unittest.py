import mxnet as mx

def verify_encode1():
    grad = [1, 0, 0]
    residual = [0, 0, 2]
    q = [[1, 1], [1, 1]]
    p = [[0, 0], [0, 0]]
    m = [[0, 0], [0, 0]]
    powersgd_encode1(grad, q, residual, m, p)

def mx_nd_array_wrapper(tensor):
    ctx = mx.gpu(0)
    return mx.nd.array(tensor, ctx=ctx, dtype='float32')

def powersgd_encode1(grad, q, residual, m, p):
    grad = mx_nd_array_wrapper(grad)
    q = mx_nd_array_wrapper(q)
    residual = mx_nd_array_wrapper(residual)
    m = mx_nd_array_wrapper(m)
    p = mx_nd_array_wrapper(p)

    true_m = mx.nd.zeros_like(m)
    (grad + residual).copyto(true_m.reshape(true_m.size)[:grad.size])
    true_p = mx.nd.zeros_like(p)
    mx.nd.dot(true_m, q, out=true_p)

    mx.nd.contrib.power_sgd_encode1(
        grad=grad,
        q=q,
        residual=residual,
        m=m,
        out=p
    )
    p.wait_to_read()
    assert mx.nd.sum((true_p - p) != 0) == 0
    m.wait_to_read()
    assert mx.nd.sum((true_m - m) != 0) == 0
    print("powersgd_encode1 tests passed!")

def verify_encode2():
    q = [[0, 0], [0, 0]]
    p = [[1, 1], [1, 1]]
    m = [[1, 0], [1, 0]]
    powersgd_encode2(p, q, m)

def verify_qr():
    N = 512
    # N = 2
    q = [[5.9604645e-07, 2.3841858e-07] for _ in range(N)]
    p = [[-2.1457672e-06, 6.5565109e-07] for _ in range(N)]
    m = [[-2.2627430e-06] * N for _ in range(N)]
    m = mx_nd_array_wrapper(m)
    p = mx_nd_array_wrapper(p)
    q = mx_nd_array_wrapper(q)
    # mx.nd.contrib.power_sgd_encode2(
    #     p=p,
    #     m=m,
    #     out=q
    # )
    powersgd_encode2(p, q, m)

def powersgd_encode2(p, q, m):
    m = mx_nd_array_wrapper(m)
    p = mx_nd_array_wrapper(p)
    q = mx_nd_array_wrapper(q)
    
    true_q = mx.nd.zeros_like(q)
    true_p = mx.nd.zeros_like(p)
    true_m = mx.nd.zeros_like(m)
    p.copyto(true_p)
    m.copyto(true_m)

    true_p_ortho, _ = mx.np.linalg.qr(true_p.as_np_ndarray())
    true_p_ortho.as_nd_ndarray().copyto(true_p)
    mx.nd.dot(true_m, true_p, transpose_a=True, out=true_q)

    mx.nd.contrib.power_sgd_encode2(
        p=p,
        m=m,
        out=q
    )
    q.wait_to_read()
    # print(true_q)
    # print(q)
    assert mx.nd.sum(abs(true_q - q) > 1e-5) == 0
    print("powersgd_encode2 tests passed!")

def verify_decode():
    r = [[7, 7], [6, 6]]
    g = [[9, 9], [8, 8]]
    q = [[0, 3], [6, 7]]
    p = [[1, 5], [12, 9]]
    m = [[1, 0], [1, 0]]
    powersgd_decode(g, q, r, m, p)

def powersgd_decode(g, q, r, m, p):
    g = mx_nd_array_wrapper(g)
    q = mx_nd_array_wrapper(q)
    r = mx_nd_array_wrapper(r)
    m = mx_nd_array_wrapper(m)
    p = mx_nd_array_wrapper(p)
    
    true_m = mx.nd.zeros_like(m)
    true_r = mx.nd.zeros_like(r)
    true_g = mx.nd.zeros_like(g)

    N = g.size

    # M = P * Q^T
    mx.nd.dot(p, q, transpose_b=True, out=true_m)

    # Residual = Grad - M
    (g - true_m).copyto(true_r)

    # G = M[:N]
    true_m[:N].copyto(true_g)

    mx.nd.contrib.power_sgd_decode(
        grad=g,
        q=q,
        residual=r,
        m=m,
        p=p
    )
    # g.wait_to_read()
    # true_g.wait_to_read()
    m.wait_to_read()
    # true_m.wait_to_read()
    # r.wait_to_read()
    # true_r.wait_to_read()
    print(true_m)
    print(m)
    assert mx.nd.sum(abs(true_m - m) > 1e-5) == 0
    assert mx.nd.sum(abs(true_r - r) > 1e-5) == 0
    assert mx.nd.sum(abs(true_g - g) > 1e-5) == 0
    print("powersgd_decode tests passed!")

if __name__ == '__main__':
    # verify_encode2()
    # verify_encode1()
    # verify_qr()
    verify_decode()

