import time

class Timer(object):
    def __init__(self, tag='time'):
        self.tag = tag

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        t = time.time() - self.start
        print('{}: {}'.format(self.tag, t))

def test_timer():
    with Timer('not end'):
        s = 0
        for i in range(1000000):
            s += i * i
        print(s)

if __name__ == '__main__':
    test_timer()