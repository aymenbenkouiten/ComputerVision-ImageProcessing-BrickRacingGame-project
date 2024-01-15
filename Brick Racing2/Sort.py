class Sort():
    @staticmethod
    def quick_sort(arr):
       if len(arr) <= 1:
           return arr
       
       pivot = arr[len(arr) // 2]
       left = [x for x in arr if x < pivot]
       middle = [x for x in arr if x == pivot]
       right = [x for x in arr if x > pivot]
       
       return Sort.quick_sort(left) + middle + Sort.quick_sort(right)