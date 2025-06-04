from slickconf import field

train = field(
    root="/mnt/ddn/seonghyeon/datasets/",
    ratios={"EleutherAI_hendrycks_math": 1.0},
    shards={
        "EleutherAI_hendrycks_math": {
            "algebra-train-1-of-1.arrayrecord": 1744,
            "counting_and_probability-train-1-of-1.arrayrecord": 771,
            "geometry-train-1-of-1.arrayrecord": 870,
            "intermediate_algebra-train-1-of-1.arrayrecord": 1295,
            "number_theory-train-1-of-1.arrayrecord": 869,
            "prealgebra-train-1-of-1.arrayrecord": 1205,
            "precalculus-train-1-of-1.arrayrecord": 746,
        }
    },
)

test = field(
    root="/mnt/ddn/seonghyeon/",
    ratios={"EleutherAI_hendrycks_math": 1.0},
    shards={
        "EleutherAI_hendrycks_math": {
            "algebra-train-1-of-1.arrayrecord-test-1-of-1.arrayrecord": 1187,
            "counting_and_probability-train-1-of-1.arrayrecord-test-1-of-1.arrayrecord": 474,
            "geometry-train-1-of-1.arrayrecord-test-1-of-1.arrayrecord": 479,
            "intermediate_algebra-train-1-of-1.arrayrecord-test-1-of-1.arrayrecord": 903,
            "number_theory-train-1-of-1.arrayrecord-test-1-of-1.arrayrecord": 540,
            "prealgebra-train-1-of-1.arrayrecord-test-1-of-1.arrayrecord": 871,
            "precalculus-train-1-of-1.arrayrecord-test-1-of-1.arrayrecord": 546,
        }
    },
)


conf = field(
    train=train,
    test=test,
)
