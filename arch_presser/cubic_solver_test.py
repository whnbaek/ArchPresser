from cubic_solver import CubicSolver


def main():
    sol = CubicSolver(-10, 5)
    #ans = sol(1, -3 , 3, -1) # test for (x-1)*3
    #ans = sol(1, 3.5, 2, -2) # test for (x+2)**2(x-0.5)
    ans = sol(1, 8.5, 9.5, -7) #test for -2 -7 0.5
    print(ans)

if __name__ == '__main__':
    main()