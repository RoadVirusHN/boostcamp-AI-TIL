import numpy as np

def mc_int(fun, low, high, smaple_size=100, repeat=10):
    int_len = np.abs(high - low)
    stat = []
    for _ in range(repeat):
        x = np.random.uniform(low=low, high=high, size=sample_size)
        fun_x = fun(x)
        int_val = int_len * np.mean(fun_x)
        stat.append(int_val)
    return np.mean(stat), np.std(stat)



def f_x(x):
    return 


print(mc_int(f_x, low=-1, hight=1,))