from agent.insight_text import remove_meta_question_phrasing


def test_chart_question_boilerplate_is_removed_without_losing_description():
    original = (
        "如图1所示，横轴为预报时效，颜色表示阈值，"
        "可回答“随时效延长，综合命中能力如何变化以及谁更优”。\n"
        "如图2所示，不同模型使用不同线型。"
    )

    cleaned = remove_meta_question_phrasing(original)

    assert "可回答" not in cleaned
    assert "综合命中能力如何变化" not in cleaned
    assert "横轴为预报时效" in cleaned
    assert "如图2所示" in cleaned
