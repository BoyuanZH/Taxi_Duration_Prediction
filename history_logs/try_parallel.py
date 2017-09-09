import multiprocessing as mp;
import time

def fn(x):
    time.sleep(1)
    print("Start {}th task.".format(x))
    return x

pool = mp.Pool(4)
l = [1, 2, 3, 4, 5, 6, 7, 8]

print("Single Thread Start...")
t0 = time.time()
for i in l:
    fn(i)
t1 = time.time()
print("Single Thread time : {} seconds".format(t1-t0))

###################

print("Paralell Start...")
t0 = time.time()
an = pool.map(fn, l)
t1 = time.time()

print("Paralell time : {} seconds".format(t1-t0))
print(an)


# def low_work(t, s):
#     print("%d %d" % (t, s))
#     time.sleep(t)
    
# def run_func(args):
#     t, s = args
#     low_work(t, s)
#     # low_work(args[0], args[1])
# if __name__ == '__main__':
#     pool = mp.Pool(4)
#     tasks = [(x, y) for x in range(4,8) for y in range(10, 14)]

#     pool.map(run_func, tasks)
#     pool.close()
#     print("main process finished")
