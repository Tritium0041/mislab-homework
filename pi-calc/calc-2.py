#用拉马努金的公式求圆周率的值

import math
import decimal
import threading


def calc(idx,start,end,result):
    s=0
    for k in range(start,end):
        s += (math.factorial(4*k)*(1103+26390*k))/(math.factorial(k)**4*396**(4*k))
    result[idx]=s
    print(idx,"finished")

def main():
    decimal.getcontext().prec = int(input("请输入你想要的圆周率的精度："))
    #多线程加速
    threads = []
    result = [0,0,0,0,0,0,0,0,0,0]
    for i in range(10):
        t = threading.Thread(target=calc,args=(i,i*200,(i+1)*200,result))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    s=sum(result)
    pi = decimal.Decimal(1)/(decimal.Decimal(s)*decimal.Decimal(2)*decimal.Decimal(math.sqrt(2))/9801)
    print(pi)
    with open("pi.txt","w") as f:
        f.write(str(pi))
        f.close()

if __name__ == '__main__':
    main()


