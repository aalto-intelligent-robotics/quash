# pylint: disable=import-outside-toplevel
from methods.common.query_api import QueryAPI
from typing import List


class AbsenceQuerier:
    def __init__(self, model: str, temp_folder: str) -> None:
        self.query_api = QueryAPI(model, temp_folder)
        self.model = model

    def get_absence_synonyms(self, query: str) -> List[str]:
        if self.model == "gpt3.5-turbo":
            from methods.prompt import absence_system_prompt_35 as absence_system_prompt
            from methods.prompt import (
                absence_synonym_prompt_35 as absence_synonym_prompt,
            )
        else:
            from methods.prompt import absence_system_prompt
            from methods.prompt import absence_synonym_prompt

        a_prompt = absence_synonym_prompt.replace("<query>", query)
        absence_synonyms_raw = self.query_api.query(absence_system_prompt, a_prompt)
        absence_synonyms = absence_synonyms_raw.split(",")
        for i, absence_synonym in enumerate(absence_synonyms):
            absence_synonyms[i] = absence_synonym.strip()
        return absence_synonyms
