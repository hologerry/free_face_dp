class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

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


def build_avg_meters(names):
    avg_meters = {}
    for name in names:
        avg_meters[name] = AverageMeter()
    return avg_meters


def update_avg_meters(value_dict, avg_meters_dict, size=1):
    for k, v in value_dict.items():
        avg_meters_dict[k].update(v, size)


def get_avg_meters_str(avg_meters_dict):
    avg_meters_str = ''
    for k, v in avg_meters_dict.items():
        avg_meters_str += f'{k}: {v.avg:.4f}  '
    return avg_meters_str