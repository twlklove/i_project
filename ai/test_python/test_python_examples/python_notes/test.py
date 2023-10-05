
def test_0():
    L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']
    r = []
    n = 3
    for i in range(n):
        r.append(L[i])
    print(r);

    print(L[0:3])
    print(L[:3])
    print(L[1:3])
    print(L[-2:])
    print(L[-2:-1])

    D = list(range(100)) 
    print(D)
    print(D[:])
    print(D[:10])
    print(D[-10:])
    print(D[10:20])
    print(D[:10:2])
    print(D[::5])

    print((0, 1, 2, 3, 4, 5)[:3])
    print('abcdef'[:3])






def main():
    test_0()

if __name__ == '__main__' :
    main()
