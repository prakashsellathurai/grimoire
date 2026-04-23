from typing import Self, Optional

class LinkedList:
    def __init__(self, val, next) -> None:
        self.val = val
        self.next = next
        
    def find(self, val):
        cur = self
        while cur is not None:
            if cur.val == val:
                return cur
            cur = cur.next
        return None
        
    def insert_after(self, target_val: int, new_val: int) -> bool:
        """Finds target_val and inserts a new node with new_val after it."""
        target_node = self.find(target_val)
        if not target_node:
            return False
        # Create new node pointing to target's original next
        new_node = LinkedList(new_val, target_node.next)
        target_node.next = new_node
        return True

    def delete_after(self, target_val: int) -> bool:
        """Finds target_val and deletes the node that follows it."""
        target_node = self.find(target_val)
        if not target_node:
            return False
        if not target_node.next:
            target_node.next = None
            return True
        # Skip the next node
        target_node.next = target_node.next.next
        return True
        
        
        