my_list = [1, 5, 2, 7, 4, 8, 9, 0]
def bubble_sort(my_list):
    for i in range(len(my_list)):
        for j in range(len(my_list) - 1):
            if my_list[j] > my_list[i]:
                my_list[j], my_list[i] = my_list[i], my_list[j]
    return my_list

print(bubble_sort(my_list))