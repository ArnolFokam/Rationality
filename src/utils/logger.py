import os
import csv


class ExperimentLogger:
    def __init__(self, log_dir, filename, header):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
        self.file.flush()

    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
