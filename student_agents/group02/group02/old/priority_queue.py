import heapq
from functools import total_ordering


# this annotation saves us some implementation work
@total_ordering
class Item(object):
    def __init__(self, insertion, priority, value):
        self.insertion = insertion
        self.priority = priority
        self.value = value

    def __lt__(self, other):
        # if the decision "self < other" can be done
        # based on the priority, do that
        if self.priority < other.priority:
            return True
        elif self.priority == other.priority:
            # in case the priorities are equal, we
            # fall back on the insertion order,
            # which establishes a total ordering
            return self.insertion < other.insertion
        return False

    def __eq__(self, other):
        return self.priority == other.priority and self.insertion == other.insertion

    def __repr__(self):
        return '({}, {}, {})'.format(self.priority, self.insertion, self.value)


class PriorityQueue(object):
    def __init__(self):
        self.insertion = 0
        self.heap = []

    def has_elements(self):
        return len(self.heap) > 0

    def put(self, priority, value):
        heapq.heappush(self.heap, Item(self.insertion, priority, value))
        self.insertion += 1

    def get(self, include_priority=False):
        item = heapq.heappop(self.heap)
        if include_priority:
            return item.priority, item.value
        else:
            return item.value

    def __iter__(self):
        return iter([item.value for item in self.heap])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ('PriorityQueue [' + ','.join((str(item.value) for item in self.heap)) + ']')

    def __len__(self):
        return len(self.heap)
