from pydantic import BaseModel
from typing import List


class EvaluatedSkill(BaseModel):
    skill: str
    relevance: int
    reason: str


class SkillEvaluation(BaseModel):
    job: str
    evaluated_skills: List[EvaluatedSkill]


class ExtractedSkills(BaseModel):
    job: str
    skills: List[str]
    required_skills: List[str]
