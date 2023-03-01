import math
import decimal
import threading


#用pi的莱布尼茨公式求pi的指定位数并写入文件，并使用多线程加速
#效果很奇怪 暂停 转calc-2


def clac(idx,start,end,result):
    tempresult=decimal.Decimal("0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001")
    for i in range(start,end):
        if i%2==0:
            tempresult+=decimal.Decimal(4.0)/decimal.Decimal(2.0*i+1.0)
        else:
            tempresult-=decimal.Decimal(4.0)/decimal.Decimal(2.0*i+1.0)
    result[idx]=decimal.Decimal(tempresult)
    print(result[idx])
    print(idx,"finished")

def main():
    decimal.getcontext().prec = 1000
    pi = decimal.Decimal(0)
    threads=[]
    result=[decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0),decimal.Decimal(0)]
    for i in range(10):
        t=threading.Thread(target=clac,args=(i,i*10000000,(i+1)*10000000,result))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    for i in result:
        pi=pi.__add__(i)
    print(pi)
    with open("pi.txt","w") as f:
        f.write(str(pi))
        f.close()

if  __name__ == '__main__':

    main()