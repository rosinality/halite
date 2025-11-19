from slickconf import field

train = field(
    root="/mnt/ddn/halite/",
    ratios={"EleutherAI_hendrycks_math": 1.0},
    shards={
        "EleutherAI_hendrycks_math": {
            "algebra-train-1-of-1.bag": 1744,
            "counting_and_probability-train-1-of-1.bag": 771,
            "geometry-train-1-of-1.bag": 870,
            "intermediate_algebra-train-1-of-1.bag": 1295,
            "number_theory-train-1-of-1.bag": 869,
            "prealgebra-train-1-of-1.bag": 1205,
            "precalculus-train-1-of-1.bag": 746,
        }
    },
)

test = field(
    root="/mnt/ddn/halite/",
    ratios={"EleutherAI_hendrycks_math": 1.0},
    shards={
        "EleutherAI_hendrycks_math": {
            "algebra-test-1-of-1.bag": 1187,
            "counting_and_probability-test-1-of-1.bag": 474,
            "geometry-test-1-of-1.bag": 479,
            "intermediate_algebra-test-1-of-1.bag": 903,
            "number_theory-test-1-of-1.bag": 540,
            "prealgebra-test-1-of-1.bag": 871,
            "precalculus-test-1-of-1.bag": 546,
        }
    },
)


conf = field(
    train=train,
    test=test,
)
