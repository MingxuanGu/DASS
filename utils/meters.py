from __future__ import absolute_import


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """
        self.val: validation score
        self.avg: the average value of the validation score = sum / count
        self.sum: the summation of all the validation values
        self.count: the number of validations. Will be increased by 1 each time the function "update()" is called.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class GroupAverageMeter(object):
    """
    Computes and stores the average and current value
    Can be considered as a group of "AverageMeter" above, but with a dictionary.
    each element in the dictionary should be {key, (value, count)}.
    Among which "count can be seen as the weight factor for the value"
    """

    def __init__(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}
    def add_key(self, key):
        self.val[key] = 0
        self.avg[key] = 0
        self.sum[key] = 0
        self.count[key] = 0
    def update(self, dic):
        for key,v in dic.items():
            if key not in self.val:
                self.add_key(key)
            value, count = v
            self.sum[key] += value*count
            self.count[key] += count
            self.avg[key] = self.sum[key] / self.count[key]
