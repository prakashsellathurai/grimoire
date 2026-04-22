from typing import Self, Optional

class LinkedList:
    def __init__(self, val, next) -> None:
        self.val = val
        self.next = next
        
    def insert(self, val) -> Self:
        if val < self.val:
            return LinkedList(val, self)
        
        if self.next is None or val < self.next.val
            self.next = LinkedList(val, self.next.val)
        else:
            self.next.insert(val)
        return self
    
    def delete(self, val: int) -> Optional[Self]:
        # Case 1: The value to delete is in the current (head) node
        if self.val == val:
            return self.next  # The next node becomes the new head
            
        # Case 2: The value is further down the list
        current = self
        while current.next is not None:
            if current.next.val == val:
                # Bypass the target node to delete it
                current.next = current.next.next
                return self
            current = current.next
            
        return self # Value not found, return original list
    
        
        
        
        