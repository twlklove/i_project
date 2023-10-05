import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def test_0():
    lines=plt.plot([1, 2, 3, 4])
    # use keyword args
    plt.setp(lines, color='r', linewidth=2.0)
    plt.ylabel('some numbers')
    plt.show()
    
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.show()
    
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    plt.axis([0, 6, 0, 20])
    plt.show()

def test_2():
    # evenly sampled time at 200ms intervals
    t = np.arange(0., 5., 0.2)
    
    # red dashes, blue squares and green triangles
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()

def test_3():
    data = {'a': np.arange(50),
            'c': np.random.randint(0, 50, 50),
            'd': np.random.randn(50)}
    data['b'] = data['a'] + 10 * np.random.randn(50)
    data['d'] = np.abs(data['d']) * 100
    
    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('entry a')
    plt.ylabel('entry b')
    plt.show()

def test_4():
    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]
    
    plt.figure(figsize=(9, 3))
    
    plt.subplot(131)
    plt.bar(names, values)
    plt.subplot(132)
    plt.scatter(names, values)
    plt.subplot(133)
    plt.plot(names, values)
    plt.suptitle('Categorical Plotting')
    plt.show()

def test_5():
    def f(t):
        return np.exp(-t) * np.cos(2*np.pi*t)
    
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    
    plt.figure()
    plt.subplot(211)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    
    plt.subplot(212)
    plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
    plt.show()

#import matplotlib.pyplot as plt
#plt.figure(1)                # the first figure
#plt.subplot(211)             # the first subplot in the first figure
#plt.plot([1, 2, 3])
#plt.subplot(212)             # the second subplot in the first figure
#plt.plot([4, 5, 6])
#
#
#plt.figure(2)                # a second figure
#plt.plot([4, 5, 6])          # creates a subplot(111) by default
#
#plt.figure(1)                # figure 1 current; subplot(212) still current
#plt.subplot(211)             # make subplot(211) in figure1 current
#plt.title('Easy as 1, 2, 3') # subplot 211 title
def test_6():
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)
    
    # the histogram of the data
    n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
    
    
    plt.xlabel('Smarts', fontsize=14, color='red')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()
    
    ax = plt.subplot(111)
    
    t = np.arange(0.0, 5.0, 0.01)
    s = np.cos(2*np.pi*t)
    line, = plt.plot(t, s, lw=2)
    
    plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 )
    
    plt.ylim(-2, 2)
    plt.show()

#################################################
# Fixing random state for reproducibility
def test_7():
    np.random.seed(19680801)
    
    # make up some data in the open interval (0, 1)
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))
    
    # plot with various axes scales
    plt.figure()
    
    # linear
    plt.subplot(221)
    plt.plot(x, y)
    plt.yscale('linear')
    plt.title('linear')
    plt.grid(True)
    
    # log
    plt.subplot(222)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)
    
    # symmetric log
    plt.subplot(223)
    plt.plot(x, y - y.mean())
    plt.yscale('symlog', linthresh=0.01)
    plt.title('symlog')
    plt.grid(True)
    
    # logit
    plt.subplot(224)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)
    # Adjust the subplot layout, because the logit one may take more space
    # than usual, due to y-tick labels like "1 - 10^{-3}"
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    
    plt.show()

def test_3d():

    # z = x^2 + y^2
    x = np.linspace(-1, 1, 101)   #num: 100 from -10 to 10
    y = np.linspace(-1, 1, 101)   #num: 100 from -10 to 10
    x, y = np.meshgrid(x, y, indexing='ij')
    z = x**2 + y**2

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    sub = fig.add_subplot(111, projection='3d')
    surf = sub.plot_surface(x, y, z, cmap=plt.cm.brg) #
    cb = fig.colorbar(surf, shrink=0.8, aspect=15)  # set color bar

    sub.set_xlabel(r'x axis')
    sub.set_ylabel(r'y axis')
    sub.set_zlabel(r'z axis')

    plt.show()

test_3d()
