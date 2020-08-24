import numpy as np

def dtw2(t,r):
    N = t.shape[1]
    M = r.shape[1]
    d = cdist(np.transpose(t),np.transpose(r),'sqeuclidean')
    D = np.zeros(shape=d.shape)
    D[0,0] = d[0,0]

    for n in range(1,N):
        D[n,0] = d[n,0] + D[n-1,0]
    for m in range(1,M):
        D[0, m] = d[0, m] + D[0, m-1]
    for n in range(1,N):
        for m in range(1,M):
            D[n,m] = d[n,m] + min([D[n-1,m],D[n-1,m-1],D[n,m-1]])

    Dist = D[N-1,M-1]
    n = N-1
    m = M-1
    k = 1
    w = np.array([N-1,M-1])
    while (n + m) != 0:
        if (n - 1) < 0:
            m = m - 1
        elif(m - 1) < 0:
            n = n - 1
        else:
            l = [D[n - 1, m], D[n, m - 1], D[n - 1, m - 1]]
            number = l.index(min(l))
            if number == 0:
                n = n - 1
            if number == 1:
                m = m - 1
            if number == 2:
                n = n - 1
                m = m - 1
        k = k + 1
        p = np.array([n,m])
        w = np.vstack((w,p))
    T = np.zeros(shape=(N, M))
    for temp_t in range(w.shape[0]):
        T[w[temp_t, 0], w[temp_t, 1]] = 1

    return Dist, T
