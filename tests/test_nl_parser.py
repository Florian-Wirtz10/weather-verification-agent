from agent.nl_parser import _build_system_prompt, _validate


def test_prompt_exposes_scope_schema_and_available_case():
    prompt = _build_system_prompt(
        [
            {
                "case_id": "case-1",
                "date": "2025-04-11",
                "init_time": "2025-04-11 02:12:00",
                "available_leads": [6, 12, 18],
            }
        ]
    )

    assert "不是实时天气问答助手" in prompt
    assert "2025-04-11" in prompt
    assert "unmatched_requests" in prompt
    assert "不得为了凑结果输出全部图" in prompt


def test_validate_uses_schema_and_does_not_default_unsupported_request():
    result = _validate(
        {
            "intent": "verify_case",
            "views": [{"type": "taylor_diagram"}],
        }
    )

    assert result["views"] == []
    assert any("taylor" in item for item in result["unmatched_requests"])
