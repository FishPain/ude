from langchain_core.messages import HumanMessage, SystemMessage
from utils import strip_json_markdown, decide_hire
from template import SkillEvaluation, ExtractedSkills
import json


def skills_extraction_node(state):
    model = state.get("model", None)
    base64_image = state.get("base64_image", None)
    prompt_text = (
        "You are a vision assistant that analyzes a single image. "
        "This image contains multiple cards. Each card either represents a skill or a job. "
        "Skill cards contain only the name of the skill. The job card contains: "
        "1. A job title at the top, and "
        "2. A list of required skills printed below the title. "
        "Your task is to extract: \n"
        "- The job title exactly as seen on the job card. \n"
        "- The list of required skills ONLY as written under the job card. \n"
        "- All skill names from the skill cards. \n"
        "You must only extract information that is clearly visible. "
        "Do NOT guess or infer required skills. Only include required skills if they appear directly on the job card. "
        "Do NOT infer skills from context or assumed relevance. "
        "Respond ONLY with valid raw JSON, with no explanations, no markdown, and no surrounding text. "
        "Use this exact JSON format: "
        '{"job": "string", "skills": ["string", ...], "required_skills": ["string", ...]}'
    )
    messages = [
        SystemMessage(
            content="You are an assistant that extracts job roles and skills from images. Only use visible evidence and do not make assumptions."
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            ]
        ),
    ]
    structured_output_parser = model.with_structured_output(ExtractedSkills)
    response = structured_output_parser.invoke(messages)
    state["extracted_skills"] = response.model_dump()
    return state


def skill_check_node(state):
    extracted_skills = state.get("extracted_skills", {})
    required_skills = set(extracted_skills.get("required_skills", []))
    all_skills = set(extracted_skills.get("skills", []))
    state["is_skill_met"] = required_skills.issubset(all_skills)
    return state


def skill_grader_node(state):
    model = state.get("model", None)
    extracted_skills = state.get("extracted_skills", [])

    relevance_prompt = (
        "You are a senior recruiter trained in industry-standard evaluation rubrics for various job roles. "
        "Given a job role and a list of user-acquired skills, your task is to evaluate the professional relevance of each skill "
        "to the specified role using standardized assessment criteria. "
        "For each skill, assign a relevance score from 1 to 10, where:\n"
        "1 = completely irrelevant,\n"
        "5 = moderately related but not essential,\n"
        "10 = highly relevant and directly applicable.\n"
        "Justify each score briefly, referencing how the skill aligns with common expectations and job responsibilities "
        "in that industry domain. Focus strictly on professional alignment — not soft traits or subjective preferences. "
        "Respond ONLY with valid raw JSON in the following format:\n"
        "{"
        '"job": "string", '
        '"evaluated_skills": ['
        '{"skill": "string", "relevance": integer (1-10), "reason": "short justification"}, ...'
        "]"
        "}\n"
        "Do not include markdown, commentary, or any extra text outside the JSON."
    )
    messages = [
        SystemMessage(content=relevance_prompt),
        HumanMessage(content=json.dumps(extracted_skills)),
    ]
    structured_output_parser = model.with_structured_output(SkillEvaluation)
    response = structured_output_parser.invoke(messages)
    state["scored_skills"] = response.model_dump()
    return state


def hire_decision_node(state):
    scored_skills = state.get("scored_skills", {})
    state["is_hired"] = decide_hire(scored_skills["evaluated_skills"])
    return state


def reject_cv_node(state):
    print("Dummy trigger bad things")
    return
