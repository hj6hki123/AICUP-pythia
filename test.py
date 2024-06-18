def f():
    yield(1)
    print('hello')       # 使用 yield
    yield(2)
    yield(3)
g = f()          # 賦值給變數 g
print(next(g))   # 1
print(next(g))   # 2
v = f()  
print(next(v))   # 3