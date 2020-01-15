class Metric(object):
    """
    class for computing average values of metric (loss or accuracy, etc.)
    """
    def __init__(self, name: str):
        self.name = name
        self.sum = 0.0
        self.n = 0

    def update(self, val:float):
        self.sum += val
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n

class MetricDict(object):
    """
    util class for handling multiple Metrics as dict
    """
    def __init__(self):
        self._metric_dict = {}

    def _initialize(self, src_dict):
        for k,v in src_dict.items():
            self._metric_dict[k] = Metric(k)

    def update(self, src_dict):
        if not self._metric_dict:
            self._initialize(src_dict)
        
        for k, v in src_dict.items():
            if k not in self._metric_dict.keys():
                raise ValueError('key value is invalid')

            self._metric_dict[k].update(v)

    @property
    def avg(self):
        avg_dict = {k: metric.avg for k, metric in self._metric_dict.items()}
        return avg_dict

if __name__ == '__main__':
    metric = Metric('test')
    metric.update(1.0)
    metric.update(2.0)
    print(metric.avg)

    metric_dict = MetricDict()
    loss_dict_01 = {'loss01':1.0, 'loss02':2.0}
    loss_dict_02 = {'loss01':2.0, 'loss02':4.0}
    metric_dict.update(loss_dict_01)
    metric_dict.update(loss_dict_02)
    print(metric_dict.avg)