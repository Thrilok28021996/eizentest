def minimum_time_to_transport(N, A, M, B):
    A.sort(reverse=True)
    B.sort(reverse=True)
    time = 0
    i = 0
    while A and i < M:
        capacity = B[i]
        while A and capacity >= A[-1]:
            capacity -= A.pop()
        time += 2
        i += 1
    if A:
        time += ((len(A) - 1) // M + 1) * 2
    return time


N = 4
A = [8, 1, 6, 9]
M = 3
B = [7, 3, 2]
print(minimum_time_to_transport(N, A, M, B))
