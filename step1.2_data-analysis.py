import os
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import constant as c


def load():
    # y = []
    # for npy in os.listdir(c.H1_DIR):
    #     data = np.load(os.path.join(c.H1_DIR, npy))[1]
    #     y.append(data)
    y = np.load("1H.npy")
    return np.asarray(y)


def draw_hist(x):
    fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))  
    #第二个参数是柱子宽一些还是窄一些，越大越窄越密  
    ax0.hist(x,100,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)  
    ##pdf概率分布图，一万个数落在某个区间内的数有多少个  
    ax0.set_title('pdf')  
    ax1.hist(x,20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)  
    #cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率  
    ax1.set_title("cdf")  
    fig.subplots_adjust(hspace=0.4)  
    plt.savefig("analysis_log.jpg")

def analysis():
    y = load()
    y = np.log(y)
    # np.save("1H.npy", y)
    print("0   ", len(y[y > 0]))
    print("5   " ,len(y[y > 5]))
    print("10  ", len(y[y > 10]))
    print("15  ", len(y[y > 15]))
    print("20  ", len(y[y > 20]))
    print("30  ", len(y[y > 30]))
    draw_hist(y)


if __name__ == "__main__":
    analysis()
