import numpy as np
import time

N = 1024

if __name__ == "__main__":
    # N ^ 2
    a = np.random.randn(N, N).astype(np.float32)
    
    # N ^ 2
    b = np.random.randn(N, N).astype(np.float32)

    flop = N * N * 2 * N

    print(f"{flop/1e9:.2f} GFLOP")

    for i in range(10):
        st = time.monotonic()

        C = a @ b

        et = time.monotonic()

        s = et - st

        print(f"{flop / s * 1e-9:.2f} GFLOPS")

    # Saving matrices to binary file
    with open("/tmp/matmul", "wb") as f:
        f.write(a.data)
        f.write(b.data)
        f.write(C.data)

