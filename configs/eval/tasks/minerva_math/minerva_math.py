from functools import partial

from slickconf import field, function

from halite.data.preprocess import SelectFeatures, Map
from halite.projects.common.template import get_render_fn
from halite.projects.eval.dataset import hf_dataset, first_n

from .eval import evaluate, record_to_input, fewshot_samples

fewshot_template = """{% for shot in fewshot %}
Problem:
{{ shot.problem }}

Solution: {{ shot.solution }}

{% endfor %}"""

task = field(
    name="minerva_math_algebra",
    dataset=hf_dataset("EleutherAI/hendrycks_math", "algebra", split="test"),
    evaluate_fn=function(evaluate),
    preprocess=[
        SelectFeatures(keys=("problem", "solution")),
        Map(function(record_to_input)),
    ],
    sampling_params=field(max_new_tokens=512, top_k=1, stop=["Problem:"]),
    fewshot=field(sampler=partial(first_n, n=4), samples=function(fewshot_samples)),
    prefix=get_render_fn(template=fewshot_template),
)

conf = field(tasks=[task])
