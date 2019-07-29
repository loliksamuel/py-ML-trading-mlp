from abc import abstractmethod


class AbstractDataSplitter(object):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def split(self, x):
        pass
