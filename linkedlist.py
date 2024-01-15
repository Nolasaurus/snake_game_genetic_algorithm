class Node:
    '''
    Linked list to store snake's head and body. Moves snake by inserting at beginning, and removing last node (unless the snake eats food)
    '''

    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        if not self.head:
            self.head = Node(data)
        else:
            new_node = Node(data)
            new_node.next = self.head
            self.head = new_node

    def to_list(self):
        nodes = []
        current = self.head
        while current:
            nodes.append(current.data)
            current = current.next
        return nodes

    def insert_at_beginning(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def remove_last_node(self):
        if self.head is None or self.head.next is None:
            self.head = None
            return
        current = self.head
        while current.next.next:
            current = current.next
        current.next = None

    def display(self):
        elems = []
        current_node = self.head
        while current_node is not None:  # Change here
            elems.append(current_node.data)
            current_node = current_node.next
        print(elems)
        
    def length(self):
        cur = self.head
        total = 0
        while cur.next is not None:
            total += 1
            cur = cur.next
        return total
