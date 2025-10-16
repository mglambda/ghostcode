# internal_testing.py
from ghostcode import types
from pydantic import BaseModel, Field
import yaml
from ghostcode.utility import show_model


class MockFile(BaseModel):
    """Used to package fake file data for testing."""

    filepath: str = "example.py"
    content: str = "def foo(bar):\n    pass"


default_file = MockFile()
default_code_response_part = types.CodeResponsePart(
    type="partial",
    filepath="example.py",
    language="python",
    original_code=default_file.content,
    new_code=default_file.content.replace("pass", "return 'Hello, world!'"),
    context_anchor="foo",
    notes=["Add hello world example to foo."],
    title="Example Code Response Part",
)


class CodeResponsePartTestData(BaseModel):
    """Pairs an example code response with file contents to serve as test data, mostly to test the handling of code parts."""

    id: str = Field(
        description="Unique but descriptive identifier for this test-data. E.g. 'python_replace_large_match_case_1'. Will be used to create a filename."
    )
    title: str = Field(description="Descriptive title for this test data pair.")

    expected_outcome: str = Field(
        default="",
        description="What is expected to happen when the program handles the given code response with the given file.",
    )

    code_response_part: types.CodeResponsePart = Field(
        default_factory=lambda: default_code_response_part,
        description="Example of a possible code response that might be received from the ghostcoder backend.",
    )

    file: MockFile = Field(
        default_factory=lambda: MockFile(),
        description="Made up file, which, paired with a code response that targets the file, can be used to test handling of code responses, perhaps editing the file.",
    )

    @staticmethod
    def from_file(filepath: str) -> "CodeResponsePartTestData":
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        return CodeResponsePartTestData(**data)

    def get_language(self) -> str:
        return self.code_response_part.language

    def to_yaml(self) -> str:
        return show_model(self)

    def filename(self) -> str:
        return f"{self.id}.yaml"


default_code_response_part_test_data = CodeResponsePartTestData(
    id="python_tiny_full_1",
    title="Full replace of function in tiny, one function file.",
    expected_outcome="foo function should get replaced.",
    code_response_part=default_code_response_part,
    file=default_file,
)
