import multiprocessing as mp
from threading import Thread
from queue import Queue
import time

def time_count(fun,n):
    time_start=time.time()
    fun(n)
    time_end=time.time()
    time_use=time_end-time_start
    print("The %s use %f s"%(fun.__name__,time_use))

def job(q,n):
    res=0
    for i in range(n):
        res+=i+i**2+i**3
    q.put(res)

def normal(n):
    result=0
    for i in range(3):
        res=0
        for i in range(n):
            res += i + i ** 2 + i ** 3
        result+=res
    return result


def multicore(n):
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q, n))
    p2 = mp.Process(target=job, args=(q, n))
    p3 =mp.Process(target=job,args=(q,n))
    # p4 = mp.Process(target=job, args=(q, n))
    # p5 = mp.Process(target=job, args=(q, n))
    # p6 = mp.Process(target=job, args=(q, n))
    processor_list=[p1,p2,p3]
    for processor in processor_list:
        processor.start()
    for processor in processor_list:
        processor.join()
    result=0
    for _ in range(len(processor_list)):
        result+=q.get()
    return result

def multi_thread(n):
    q=Queue()
    Td1=Thread(target=job,args=(q,n))
    Td2=Thread(target=job,args=(q,n))
    Td3 = Thread(target=job, args=(q, n))
    Td1.start()
    Td2.start()
    Td3.start()
    Td1.join()
    Td2.join()
    Td3.join()
    res1=q.get()
    res2=q.get()
    res3=q.get()
    result=res1+res2+res3
    return result

def job_pool(x):
    return x**2

def process_pool_test():
    pool=mp.Pool()
    res=pool.map(job_pool,range(1000000000))


if __name__=="__main__":
    process_pool_test()
