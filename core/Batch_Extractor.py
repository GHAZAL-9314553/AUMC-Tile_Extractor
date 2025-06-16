import os
from multiprocessing import Process, Queue, cpu_count
from typing import List, Type

from patch_extractor import BasePatchExtractor


def worker_main(task_queue: Queue):
    while not task_queue.empty():
        extractor: BasePatchExtractor
        save_dir: str
        try:
            extractor, save_dir = task_queue.get_nowait()
            extractor.extract(save_dir)
        except Exception as e:
            print(f"[Worker Error] {e}")


class BatchExtractor:
    def __init__(
        self,
        extractors: List[BasePatchExtractor],
        save_dirs: List[str],
        num_workers: int = cpu_count(),
    ):
        assert len(extractors) == len(save_dirs), "Mismatch between extractors and save_dirs"
        self.extractors = extractors
        self.save_dirs = save_dirs
        self.num_workers = num_workers

    def run(self):
        task_queue = Queue()
        for extractor, save_dir in zip(self.extractors, self.save_dirs):
            task_queue.put((extractor, save_dir))

        workers = [Process(target=worker_main, args=(task_queue,)) for _ in range(self.num_workers)]

        for w in workers:
            w.start()
        for w in workers:
            w.join()