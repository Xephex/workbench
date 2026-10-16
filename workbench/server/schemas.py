from __future__ import annotations
import base64
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

class Input(BaseModel):
    model_config = ConfigDict(extra='ignore', strict=True)

class TextPart(Input):
    type: Literal['text']
    text: str

class ImageURL(Input):
    url: str
    detail: Literal['auto', 'low', 'high'] = 'auto'
    @field_validator('url')
    @classmethod
    def local_image(cls, url):
        head, sep, data = url.partition(',')
        if not sep or head not in ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'):
            raise ValueError('Images must be inline PNG, JPEG or WebP base64 data URLs')
        if len(data) > 28_000_000:
            raise ValueError('Image exceeds 20 MB')
        base64.b64decode(data, validate=True)
        return url

class ImagePart(Input):
    type: Literal['image_url']
    image_url: ImageURL

class FunctionCall(Input):
    name: str
    arguments: str

class ToolCall(Input):
    id: str
    type: Literal['function'] = 'function'
    function: FunctionCall

class Message(Input):
    role: Literal['system', 'developer', 'user', 'assistant', 'tool', 'function']
    content: str | list[TextPart | ImagePart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    @model_validator(mode='after')
    def valid(self):
        if self.role == 'tool' and not self.tool_call_id:
            raise ValueError('Tool messages need tool_call_id')
        if self.content is None and not (self.role == 'assistant' and self.tool_calls):
            raise ValueError('Content is required')
        if self.tool_calls and self.role != 'assistant':
            raise ValueError('Only assistant messages contain tool_calls')
        if isinstance(self.content, list) and any(isinstance(x, ImagePart) for x in self.content) and self.role != 'user':
            raise ValueError('Image parts require the user role')
        return self

class Function(Input):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict = Field(default_factory=lambda: {'type': 'object', 'properties': {}})
    strict: bool | None = None

class Tool(Input):
    type: Literal['function'] = 'function'
    function: Function

class FunctionName(Input):
    name: str

class NamedChoice(Input):
    type: Literal['function']
    function: FunctionName

class SchemaSpec(Input):
    name: str
    schema_: dict = Field(alias='schema')
    strict: bool = True

class ResponseFormat(Input):
    type: Literal['text', 'json_object', 'json_schema']
    json_schema: SchemaSpec | None = None
    @model_validator(mode='after')
    def valid(self):
        if self.type == 'json_schema' and self.json_schema is None:
            raise ValueError('json_schema requires a schema')
        return self

class StreamOptions(Input):
    include_usage: bool = False

class ChatRequest(Input):
    model: str | None = None
    messages: list[Message] = Field(min_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[Tool] | None = None
    tool_choice: Literal['auto', 'none', 'required'] | NamedChoice | None = None
    response_format: ResponseFormat | None = None
    max_tokens: int = Field(default=1024, ge=1, le=131072)
    temperature: float = Field(default=.7, ge=0, le=2)
    top_p: float = Field(default=.95, gt=0, le=1)
    seed: int | None = None
    stop: str | list[str] | None = None
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    think: bool | None = None
    n: Literal[1] = 1
    @model_validator(mode='after')
    def valid(self):
        if self.tool_choice not in (None, 'none') and not self.tools:
            raise ValueError('tool_choice requires tools')
        if isinstance(self.tool_choice, NamedChoice):
            if self.tool_choice.function.name not in {t.function.name for t in self.tools or []}:
                raise ValueError('Named tool_choice is not declared in tools')
        if self.tools and self.response_format and self.response_format.type != 'text':
            raise ValueError('Use tools or response_format in a request, not both')
        return self
    def payload(self):
        result = self.model_dump(exclude_none=True, by_alias=True)
        for message in result['messages']:
            content = message.get('content')
            if isinstance(content, list) and all(p['type'] == 'text' for p in content):
                message['content'] = ''.join(p['text'] for p in content)
        return result

class EmbeddingRequest(Input):
    model: str | None = None
    input: str | list[str]
    encoding_format: Literal['float', 'base64'] = 'float'
    @model_validator(mode='after')
    def valid(self):
        if self.input == '' or self.input == [] or (isinstance(self.input, list) and any(not t for t in self.input)):
            raise ValueError('Embedding input must contain nonempty text')
        return self

class Document(Input):
    text: str

class RerankRequest(Input):
    model: str | None = None
    query: str = Field(min_length=1)
    documents: list[str | Document] = Field(min_length=1)
    top_n: int | None = Field(default=None, ge=1)
    return_documents: bool = False
