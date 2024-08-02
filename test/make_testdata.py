import json
import os
import random

from array_record.python.array_record_module import ArrayRecordWriter

if __name__ == "__main__":
    path_a = "testdata/dataset_a"
    path_b = "testdata/dataset_b"
    os.makedirs(path_a, exist_ok=True)
    os.makedirs(path_b, exist_ok=True)

    writer_a = ArrayRecordWriter(
        os.path.join(path_a, "data-1-of-1.arrayrecord"), "group_size:1"
    )
    writer_b = ArrayRecordWriter(
        os.path.join(path_b, "data-1-of-1.arrayrecord"), "group_size:1"
    )

    for i in range(1000):
        length = random.randrange(1, 100)
        data = {"text": " ".join([str(i)] * (i % 100 + 1))}
        data = json.dumps(data).encode("utf-8")
        writer_a.write(data)

        length = random.randrange(1, 100)
        data = {"text": " ".join([str(i + 1000)] * (i % 100 + 1))}
        data = json.dumps(data).encode("utf-8")
        writer_b.write(data)

    writer_a.close()
    writer_b.close()
