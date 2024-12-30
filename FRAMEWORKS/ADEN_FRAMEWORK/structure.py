from collections import deque
class Stack:
    """Represents a stack data structure."""
    def __init__(self):
        self.items = deque()

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        return None

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)


class Heap:
    """Represents a basic heap data structure (using a list and basic functions)."""
    def __init__(self):
        self.items = []

    def insert(self, item):
        self.items.append(item)
        self._heapify_up(len(self.items) - 1)

    def pop(self):
        if not self.is_empty():
           self._swap(0, len(self.items) - 1)
           popped_item = self.items.pop()
           self._heapify_down(0)
           return popped_item
        return None


    def _heapify_up(self, index):
        while index > 0:
            parent_index = (index-1) // 2
            if self.items[index] > self.items[parent_index]:
                self._swap(index, parent_index)
                index = parent_index
            else:
                break


    def _heapify_down(self, index):
        while True:
            left_child = 2 * index + 1
            right_child = 2 * index + 2
            largest = index
            if left_child < len(self.items) and self.items[left_child] > self.items[largest]:
                largest = left_child
            if right_child < len(self.items) and self.items[right_child] > self.items[largest]:
                largest = right_child
            if largest != index:
                self._swap(index, largest)
                index = largest
            else:
                break

    def _swap(self, i, j):
        self.items[i], self.items[j] = self.items[j], self.items[i]

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        return None

class Queue:
    """Represents a queue data structure."""
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.popleft()
        return None

    def peek(self):
        if not self.is_empty():
           return self.items[0]
        return None

    def is_empty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)

class Funnels:
     """Represents two funnels for input."""
     def __init__(self):
       self.top = []
       self.bottom = []

     def push_top(self, item):
          self.top.append(item)

     def push_bottom(self, item):
         self.bottom.append(item)

     def pop_top(self):
         if self.top:
              return self.top.pop()
         return None

     def pop_bottom(self):
         if self.bottom:
             return self.bottom.pop()
         return None

     def is_empty(self):
        return not (self.top or self.bottom)
     def size(self):
        return len(self.top) + len(self.bottom)
class NeutralZone:
     """Represents a neutral zone where data converges."""
     def __init__(self):
       self.items = []

     def add_to_zone(self, item):
         self.items.append(item)

     def pop_from_zone(self):
          if self.items:
             return self.items.pop()
          return None

     def peek(self):
        if not self.items:
            return None
        return self.items[-1]
     def is_empty(self):
        return len(self.items) == 0
     def size(self):
         return len(self.items)

