from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin

@dataclass
class ActivationPatchingSamples(DataClassJsonMixin):
    prompt_template: str
    common_entity: str
    clean_entity: str
    patched_entity: str
    clean_answer: str | None = None
    patched_answer: str | None = None
    patched_answer_toks: list[int] | None = None

    def __str__(self):
        return f'{self.common_entity} | {self.clean_entity} => "{self.clean_answer}" | \
            {self.patched_entity} => "{self.patched_answer}"'