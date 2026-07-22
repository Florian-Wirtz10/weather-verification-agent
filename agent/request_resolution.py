"""Resolve a validated request against deterministic product/data capabilities."""

from __future__ import annotations

from typing import Any


def resolve_request(
    request: dict[str, Any],
    available_cases: list[dict[str, Any]],
    *,
    selected_case_id: str | None = None,
) -> dict[str, Any]:
    """Return a non-LLM status and, when ready, an auditable execution plan."""
    intent = request["intent"]
    if intent == "out_of_scope":
        return _decision(
            "out_of_scope",
            "当前产品只检验已接入的风雷预报个例，不提供实时或未来天气问答。",
            request,
        )
    if intent == "show_capabilities":
        return _decision(
            "show_capabilities",
            "支持反射率阈值、预报时效范围、CSI/POD/FAR/Bias/ETS曲线、"
            "预报率—观测率关系图、指定时效的预报实况对比与误差图，以及可选结果解读。",
            request,
        )
    if intent == "unknown":
        return _decision(
            "needs_clarification", "没有足够信息可靠执行，请换一种更明确的说法。", request
        )
    if intent == "explain_last_result":
        return _decision(
            "needs_clarification",
            "Phase 3 暂不继承上一轮对话状态，请重新给出完整检验要求；"
            "每次成功检验都会自动生成结果解读。",
            request,
        )

    if not available_cases:
        return _decision("no_data", "当前没有可用且时间严格匹配的风雷/雷达个例。", request)

    requested_date = request["filters"]["requested_date"]
    candidates = available_cases
    if requested_date is not None:
        candidates = [case for case in candidates if case.get("date") == requested_date]
        if not candidates:
            dates = sorted({str(case.get("date")) for case in available_cases})
            return _decision(
                "no_data",
                f"当前没有 {requested_date} 的个例；可用日期：{', '.join(dates)}。",
                request,
            )

    selected_case = None
    if selected_case_id is not None:
        selected_case = next(
            (case for case in candidates if case.get("case_id") == selected_case_id), None
        )
    if selected_case is None and len(candidates) == 1:
        selected_case = candidates[0]
    if selected_case is None:
        choices = ", ".join(str(case.get("case_id")) for case in candidates)
        return _decision(
            "needs_clarification", f"存在多个可用起报，请明确选择：{choices}。", request
        )

    unmatched = request["unmatched_requests"]
    views = request["views"]
    if unmatched and not views:
        return _decision(
            "unsupported",
            "请求的功能当前尚未实现：" + "；".join(unmatched),
            request,
            case=selected_case,
            unmatched=unmatched,
        )
    status = "partial" if unmatched else "ready"
    message = (
        "将只执行已支持的部分；未完成：" + "；".join(unmatched)
        if unmatched
        else "请求已通过能力和数据校验。"
    )
    return _decision(
        status,
        message,
        request,
        case=selected_case,
        unmatched=unmatched,
        execution={
            "init_time": selected_case["init_time"],
            "thresholds": request["filters"]["thresholds_dbz"],
            "lead_range": request["filters"]["lead_range_minutes"],
            "views": views,
            "insight": request["insight"],
            "presentation": request.get(
                "presentation", {"layout": "auto", "strategy": "auto", "metric": None}
            ),
        },
    )


def _decision(status, message, request, **extra):
    return {"status": status, "message": message, "request": request, **extra}
