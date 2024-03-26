from queue import PriorityQueue
import Series_Q
import utils
import LowerBounds

class KNNSearch:
    def __init__(self, root_node_of_index, K):
        self.queue = PriorityQueue()
        self.temp = []
        self.result = []
        self.K = K
        self.queue.put((0, root_node_of_index))

    def search(self, Q):
        while not self.queue.empty():
            top = self.queue.get()
            #print(top)
            if isinstance(top[1], Series_Q.Series_Q):  # Check if top is a PAA point
                C = top[1]
                dist = top[0]
                self.temp.append((dist, C))
                self.temp.sort(key=lambda x: x[0])  # Sort temp by distance
                self.move_to_result(Q)
                if len(self.result) == self.K:
                    return self.result
            elif top[1].is_leaf == True:  # Check if top is a leaf node
                for C in top[1].database:
                    bound = LowerBounds.LowerBounds(Q, C)
                    lb_paa = bound.LB_PAA()
                    self.queue.put((lb_paa, C))
            elif top[1].is_leaf == False:  # Check if top is a non-leaf node
                for child_node in top[1].children:
                    L = child_node.mbrs.T[0]
                    H = child_node.mbrs.T[1]
                    mindist = utils.MINDIST(Q.Q, L, H)
                    self.queue.put((mindist, child_node))


    def move_to_result(self, Q):
        i = 0
        while i < len(self.temp):
            dist, C = self.temp[i]
            bound = LowerBounds.LowerBounds(Q, C)
            lb_paa = bound.LB_PAA()
            if lb_paa <= dist:
                self.temp.pop(i)
                self.result.append(C)
            else:
                i += 1
