#!/usr/bin/env python
# test.py
# Here I test out various sorting algorithms
import random
import time

# Example data

animals = "dog cat mule horse whale chimp spider rat bluejay".split(" ")
numbers = [random.randint(0, 2**16) for _ in range(0, 100)]
special = ["aaa", "aaa", "aab", "aaa", "aaa", "aaa"] * 100
empty_list = []
lists = [animals, numbers, empty_list]


# and here are various sorting algorithms
def merge_sort(items):
    if len(items) <= 1:
        return items

    mid = len(items) // 2
    left = items[:mid]
    right = items[mid:]

    left = merge_sort(left)
    right = merge_sort(right)

    return merge(left, right)

def merge(left, right):
    merged = []
    left_idx, right_idx = 0, 0

    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] <= right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1

    merged.extend(left[left_idx:])
    merged.extend(right[right_idx:])
    return merged

def quick_sort(items):
    if len(items) <= 1:
        return items

    pivot = items[len(items) // 2]
    left = [x for x in items if x < pivot]
    middle = [x for x in items if x == pivot]
    right = [x for x in items if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

def insertion_sort(items):
    for i in range(1, len(items)):
        key = items[i]
        j = i - 1
        while j >= 0 and key < items[j]:
            items[j + 1] = items[j]
            j -= 1
        items[j + 1] = key
    return items

def bubble_sort(items):
    n = len(items)
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            # Traverse the list from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if items[j] > items[j+1]:
                items[j], items[j+1] = items[j+1], items[j]
    return items

def selection_sort(items):
    n = len(items)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if items[j] < items[min_idx]:
                min_idx = j
        items[i], items[min_idx] = items[min_idx], items[i]
    return items

def heapify(items, n, i):
    largest = i  # Initialize largest as root
    left = 2 * i + 1
    right = 2 * i + 2

    # See if left child of root exists and is greater than root
    if left < n and items[left] > items[largest]:
        largest = left

    # See if right child of root exists and is greater than root
    if right < n and items[right] > items[largest]:
        largest = right

    # Change root, if needed
    if largest != i:
        items[i], items[largest] = items[largest], items[i]  # swap
        heapify(items, n, largest)

def heap_sort(items):
    n = len(items)

    # Build a maxheap
    for i in range(n // 2 - 1, -1, -1):
        heapify(items, n, i)

    # One by one extract elements
    for i in range(n - 1, 0, -1):
        items[i], items[0] = items[0], items[i]  # swap
        heapify(items, i, 0)
    return items

def main():
    print("--- 🧪 Testing Sorting Algorithms 🧪 ---")
    for i, original_list in enumerate(lists):
        print(f"\n✨ Testing list {i+1}: ✨")
        print(f"  Original: {original_list} 📋")

        # Test merge_sort
        list_copy_merge = list(original_list) # Create a copy to avoid modifying the original
        start_time_merge = time.perf_counter()
        sorted_list_merge = merge_sort(list_copy_merge)
        end_time_merge = time.perf_counter()
        print(f"  Merge Sort: {sorted_list_merge} 🚀")
        print(f"    Time: {end_time_merge - start_time_merge:.6f} seconds ⏱️")

        # Test bubble_sort
        list_copy_bubble = list(original_list) # Create another copy
        start_time_bubble = time.perf_counter()
        sorted_list_bubble = bubble_sort(list_copy_bubble)
        end_time_bubble = time.perf_counter()
        print(f"  Bubble Sort: {sorted_list_bubble} 🎈")
        print(f"    Time: {end_time_bubble - start_time_bubble:.6f} seconds ⏱️")

        # Test quick_sort
        list_copy_quick = list(original_list)
        start_time_quick = time.perf_counter()
        sorted_list_quick = quick_sort(list_copy_quick)
        end_time_quick = time.perf_counter()
        print(f"  Quick Sort: {sorted_list_quick} ⚡")
        print(f"    Time: {end_time_quick - start_time_quick:.6f} seconds ⏱️")

        # Test insertion_sort
        list_copy_insertion = list(original_list)
        start_time_insertion = time.perf_counter()
        sorted_list_insertion = insertion_sort(list_copy_insertion)
        end_time_insertion = time.perf_counter()
        print(f"  Insertion Sort: {sorted_list_insertion} ➡️")
        print(f"    Time: {end_time_insertion - start_time_insertion:.6f} seconds ⏱️")

        # Test selection_sort
        list_copy_selection = list(original_list)
        start_time_selection = time.perf_counter()
        sorted_list_selection = selection_sort(list_copy_selection)
        end_time_selection = time.perf_counter()
        print(f"  Selection Sort: {sorted_list_selection} 🎯")
        print(f"    Time: {end_time_selection - start_time_selection:.6f} seconds ⏱️")

        # Test heap_sort
        list_copy_heap = list(original_list)
        start_time_heap = time.perf_counter()
        sorted_list_heap = heap_sort(list_copy_heap)
        end_time_heap = time.perf_counter()
        print(f"  Heap Sort: {sorted_list_heap} ⛰️")
        print(f"    Time: {end_time_heap - start_time_heap:.6f} seconds ⏱️")

if __name__ == "__main__":
    main()
