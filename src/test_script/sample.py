from utils import calc_time
from abc import ABC, abstractmethod
import functools
import time


"""
@calc_timeのついたabstractmethodはを継承する
"""


class BaseProcessor(ABC):
    """
    BaseProcessManagerは、Processorを継承したクラスを持つ
    そのため、processメソッドを持つ
    """

    @abstractmethod
    def process(self):
        pass

    def __new__(cls, *args, **kwargs):
        cls.process = calc_time(cls.process)
        return super().__new__(cls)


class Processor(BaseProcessor):
    """
    Processorは、BaseProcessManagerの中でつなぎ合わせるもの
    """

    def __init__(self):
        pass

    def process(self):
        super().process()
        count = 0
        for i in range(10**8):
            count += 1


@calc_time
def example_function():
    count = 0
    for i in range(10**8):
        count += 1
    return count


processor = Processor()
processor.process()
