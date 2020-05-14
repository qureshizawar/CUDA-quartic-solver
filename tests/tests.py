import numpy as np
import QuarticSolver

def test1():
    eps = 1e-5
    correct_minimum = -2

    N = 1<<9
    A = np.ones(N)*2
    B = np.ones(N)*-4
    C = np.ones(N)*-22
    D = np.ones(N)*24
    E = np.ones(N)*2

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

    N = 1<<13
    A = np.ones(N)*2
    B = np.ones(N)*-4
    C = np.ones(N)*-22
    D = np.ones(N)*24
    E = np.ones(N)*2

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

    N = 1<<20
    A = np.ones(N)*2
    B = np.ones(N)*-4
    C = np.ones(N)*-22
    D = np.ones(N)*24
    E = np.ones(N)*2

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

def test2():
    eps = 1e-5
    correct_minimum = -0.5688

    N = 1<<9
    A = np.ones(N)*14
    B = np.ones(N)*-11
    C = np.ones(N)*51
    D = np.ones(N)*79
    E = np.ones(N)*1

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

    N = 1<<13
    A = np.ones(N)*14
    B = np.ones(N)*-11
    C = np.ones(N)*51
    D = np.ones(N)*79
    E = np.ones(N)*1

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

    N = 1<<20
    A = np.ones(N)*14
    B = np.ones(N)*-11
    C = np.ones(N)*51
    D = np.ones(N)*79
    E = np.ones(N)*1

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

def test3():
    eps = 1e-5
    correct_minimum = -21.75

    N = 1<<9
    A = np.ones(N)*3
    B = np.ones(N)*87
    C = np.ones(N)*0
    D = np.ones(N)*0
    E = np.ones(N)*0

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

    N = 1<<13
    A = np.ones(N)*3
    B = np.ones(N)*87
    C = np.ones(N)*0
    D = np.ones(N)*0
    E = np.ones(N)*0

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps

    N = 1<<20
    A = np.ones(N)*3
    B = np.ones(N)*87
    C = np.ones(N)*0
    D = np.ones(N)*0
    E = np.ones(N)*0

    minimum = QuarticSolver.QuarticMinimum(A,B,C,D,E,True)
    res = np.sum(minimum)/N
    assert abs(res-correct_minimum)<eps


if __name__=='__main__':

    QuarticSolver.dry_run(1<<20)

    test1()
    test2()
    test3()
    print('all test passed!')
