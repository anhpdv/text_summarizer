import timeit


def my_function():
    if 1 == 1:
        return True
    else:
        False


execution_time = timeit.timeit(my_function, number=1)
print("Thời gian chạy của hàm là:", execution_time, "giây")

# Thời gian thực thi: Độ phức tạp thời gian thực thi của mỗi hàm,
# tức là thời gian nó mất để thực hiện công việc.
# Hàm nào mất nhiều thời gian hơn để chạy có thể được xem là phức tạp hơn.