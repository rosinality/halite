from slickconf import field

conf = field(
    root="/mnt/ddn/seonghyeon/datasets/",
    ratios={
        "agentica-org_DeepScaleR-Preview-Dataset": 1.0
    },
    shards={
        "agentica-org_DeepScaleR-Preview-Dataset": {
            "train-1-of-1.arrayrecord": 40315
        }
    },
)
