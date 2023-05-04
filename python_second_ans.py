def find_consecutive_odd_numbers_sum(n):
    count = 0
    for i in range(1, n + 1, 2):
        sum = 0
        for j in range(i, n + 1, 2):
            sum += j
            if sum == n:
                count += 1
                print(f"Set {count}: {[k for k in range(i, j+1, 2)]}")
                break
            elif sum > n:
                break
    print(f"Total number of sets: {count}")


n = 11
print(f"Odd numbers up to {n}: {[i for i in range(1, n+1, 2)]}")
find_consecutive_odd_numbers_sum(n)
