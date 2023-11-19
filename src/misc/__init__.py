class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def timer(func):
    def func_wrapper(*args, **kwargs):
        from datetime import datetime
        t0 = datetime.now()
        ret = func(*args, **kwargs)
        t1 = datetime.now()
        t = t1 - t0
        print(f"{func.__name__} cost time: {str(t)}")
        return ret
    return func_wrapper

