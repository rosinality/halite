import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from halite.transformers.infer.types import InferenceResult


def nowstamp():
    return int(datetime.now().timestamp())


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = False


class CompletionsRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[list[int], list[list[int]], str, list[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    metadata: Optional[dict] = None

    # extra fields to get sglang benchmarking script to work
    ignore_eos: bool = False

    class Config:
        extra = "forbid"


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # use alias to workaround pydantic conflict
    schema_: Optional[dict[str, object]] = Field(alias="schema", default=None)
    strict: Optional[bool] = False


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None


class BatchCreationRequest(BaseModel):
    """Request model for creating a batch"""

    input_file_id: str = Field(
        description="The ID of an uploaded file that contains requests for the new batch"
    )
    endpoint: str = Field(
        description="The endpoint to be used for all requests in the batch"
    )
    completion_window: str = Field(
        description="The time frame within which the batch should be processed"
    )
    metadata: Optional[dict[str, str]] = Field(default=None)


@dataclass
class SamplingParams:
    temperature: float
    top_p: float


@dataclass
class TokasaurusRequest:
    id: str
    input_ids: list[int]
    max_num_tokens: int
    sampling_params: SamplingParams
    stop: list[str]
    n: int
    ignore_eos: bool
    created_timestamp: float = field(default_factory=time.time)


@dataclass
class SubmittedRequest:
    request: TokasaurusRequest
    engine_index: int

    event: asyncio.Event = field(default_factory=asyncio.Event)
    request_output: InferenceResult | None = None


class BatchFileLine(BaseModel):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/completions", "/v1/chat/completions"]
    body: dict


@dataclass
class SubmittedBatchItem:
    line: BatchFileLine
    user_req: CompletionsRequest
    submitted_req: SubmittedRequest


@dataclass
class SubmittedBatch:
    id: str
    creation_request: BatchCreationRequest
    items: list[SubmittedBatchItem]
    task: asyncio.Task
    created_at: int = field(default_factory=nowstamp)
    output_file: None = None


@dataclass
class RequestError:
    error: str


@dataclass
class CancelledRequest:
    req_id: str


CommandsFromServer = TokasaurusRequest | CancelledRequest
