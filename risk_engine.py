import pandas as pd


SEVERITY_WEIGHTS = {"High": 18, "Medium": 10, "Low": 4}
TYPE_WEIGHTS = {
    "Bank": 1.0,
    "Capital": 1.08,
    "Overseas Bank": 0.95,
    "Asset Management": 0.9,
}


def _safe_pct_change(current, previous):
    if previous == 0:
        return 0.0
    return round(((current - previous) / previous) * 100, 1)


def _delta_pp(current, previous):
    return round(current - previous, 2)


def _latest_and_previous(group: pd.DataFrame):
    latest = group.iloc[-1]
    previous = group.iloc[-2] if len(group) > 1 else latest
    return latest, previous


def _previous_n_average(group: pd.DataFrame, column: str, n: int = 3):
    if len(group) <= 1:
        return float(group.iloc[-1][column])
    previous_rows = group.iloc[:-1].tail(n)
    if previous_rows.empty:
        return float(group.iloc[-1][column])
    return round(float(previous_rows[column].mean()), 2)


def _score_component(change_pct, strong_threshold, medium_threshold, high_score, medium_score):
    if change_pct >= strong_threshold:
        return high_score
    if change_pct >= medium_threshold:
        return medium_score
    return 0


def _build_driver_summary(company_drivers: pd.DataFrame, direction: str | None = None, top_n: int = 3):
    if company_drivers.empty:
        return "주요 드라이버 정보 없음"
    target = company_drivers.copy()
    if direction == "positive":
        target = target[target["direction"] == "positive"]
    elif direction == "negative":
        target = target[target["direction"] == "negative"]
    if target.empty:
        return "주요 드라이버 정보 없음"
    target = target.assign(abs_contribution=target["contribution_bps"].abs()).sort_values("abs_contribution", ascending=False)
    items = [f"{row['driver_name']}({abs(int(row['contribution_bps']))}bp)" for _, row in target.head(top_n).iterrows()]
    return ", ".join(items)


def _executive_headline(change_pp: float, positive_summary: str, negative_summary: str):
    if change_pp < 0:
        return f"연체율은 {positive_summary} 영향으로 개선되었습니다."
    if change_pp > 0:
        return f"연체율은 {negative_summary} 영향으로 상승했습니다."
    return "연체율은 전월과 유사한 수준을 유지했습니다."


def calculate_company_risk(
    metrics_df: pd.DataFrame,
    logs_df: pd.DataFrame | None = None,
    drivers_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metrics_df = metrics_df.sort_values(["company_name", "date"]).copy()
    rows = []

    latest_month = metrics_df["date"].max().to_period("M")
    recent_logs = pd.DataFrame()
    recent_drivers = pd.DataFrame()

    if logs_df is not None and not logs_df.empty:
        recent_logs = logs_df[logs_df["date"].dt.to_period("M") == latest_month].copy()
    if drivers_df is not None and not drivers_df.empty:
        recent_drivers = drivers_df[drivers_df["date"].dt.to_period("M") == latest_month].copy()

    for company, group in metrics_df.groupby("company_name"):
        group = group.sort_values("date")
        latest, previous = _latest_and_previous(group)
        company_type = latest["company_type"]
        type_weight = TYPE_WEIGHTS.get(company_type, 1.0)

        delinquency_change_pct = _safe_pct_change(latest["delinquency_rate"], previous["delinquency_rate"])
        complaints_change_pct = _safe_pct_change(latest["complaints"], previous["complaints"])
        abnormal_events_change_pct = _safe_pct_change(latest["abnormal_events"], previous["abnormal_events"])
        real_estate_change_pct = _safe_pct_change(latest["exposure_real_estate"], previous["exposure_real_estate"])
        sme_change_pct = _safe_pct_change(latest["exposure_sme"], previous["exposure_sme"])

        delinquency_change_pp = _delta_pp(latest["delinquency_rate"], previous["delinquency_rate"])
        trailing_3m_avg = _previous_n_average(group, "delinquency_rate", 3)
        vs_3m_avg_pp = _delta_pp(latest["delinquency_rate"], trailing_3m_avg)
        vs_3m_avg_pct = _safe_pct_change(latest["delinquency_rate"], trailing_3m_avg) if trailing_3m_avg != 0 else 0.0

        component_scores = {
            "신용리스크": _score_component(delinquency_change_pct, 20, 10, 34, 20),
            "민원/소비자보호": _score_component(complaints_change_pct, 18, 10, 18, 10),
            "운영리스크": _score_component(abnormal_events_change_pct, 20, 10, 22, 12),
            "부동산 집중": _score_component(real_estate_change_pct, 15, 8, 12, 7),
            "SME 집중": _score_component(sme_change_pct, 12, 6, 10, 5),
            "3개월 평균 대비 압력": _score_component(vs_3m_avg_pct, 15, 8, 12, 6),
        }

        log_risk_score = 0
        log_messages = []
        if not recent_logs.empty:
            company_logs = recent_logs[recent_logs["company_name"] == company]
            for _, log_row in company_logs.iterrows():
                log_risk_score += SEVERITY_WEIGHTS.get(log_row["severity"], 0)
                log_messages.append(f"로그:{log_row['issue_type']}({log_row['severity']})")

        positive_summary = "개선 드라이버 정보 없음"
        negative_summary = "악화 드라이버 정보 없음"
        if not recent_drivers.empty:
            company_drivers = recent_drivers[recent_drivers["company_name"] == company]
            positive_summary = _build_driver_summary(company_drivers, direction="positive")
            negative_summary = _build_driver_summary(company_drivers, direction="negative")

        base_score = sum(component_scores.values())
        weighted_score = round(min((base_score + log_risk_score) * type_weight, 100), 1)

        if weighted_score >= 75:
            risk_level = "High"
        elif weighted_score >= 45:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        drivers = [name for name, score in component_scores.items() if score > 0]
        drivers.extend(log_messages)
        if not drivers:
            drivers = ["주요 이상 없음"]

        rows.append(
            {
                "company_name": company,
                "company_type": company_type,
                "risk_score": weighted_score,
                "risk_level": risk_level,
                "delinquency_change_pct": delinquency_change_pct,
                "delinquency_change_pp": delinquency_change_pp,
                "complaints_change_pct": complaints_change_pct,
                "abnormal_events_change_pct": abnormal_events_change_pct,
                "real_estate_change_pct": real_estate_change_pct,
                "sme_change_pct": sme_change_pct,
                "trailing_3m_avg": trailing_3m_avg,
                "vs_3m_avg_pp": vs_3m_avg_pp,
                "vs_3m_avg_pct": vs_3m_avg_pct,
                "latest_delinquency_rate": latest["delinquency_rate"],
                "previous_delinquency_rate": previous["delinquency_rate"],
                "latest_complaints": latest["complaints"],
                "latest_abnormal_events": latest["abnormal_events"],
                "latest_exposure_real_estate": latest["exposure_real_estate"],
                "latest_exposure_sme": latest["exposure_sme"],
                "credit_score": component_scores["신용리스크"],
                "complaint_score": component_scores["민원/소비자보호"],
                "operational_score": component_scores["운영리스크"],
                "real_estate_score": component_scores["부동산 집중"],
                "sme_score": component_scores["SME 집중"],
                "trend_pressure_score": component_scores["3개월 평균 대비 압력"],
                "log_risk_score": log_risk_score,
                "top_drivers": "|".join(drivers[:5]),
                "top_drivers_text": ", ".join(drivers[:3]),
                "positive_driver_summary": positive_summary,
                "negative_driver_summary": negative_summary,
                "executive_headline": _executive_headline(delinquency_change_pp, positive_summary, negative_summary),
            }
        )

    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def detect_alerts(metrics_df: pd.DataFrame, logs_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = metrics_df.sort_values(["company_name", "date"]).copy()
    alerts = []

    for company, group in metrics_df.groupby("company_name"):
        latest, previous = _latest_and_previous(group.sort_values("date"))

        delinquency_change_pct = _safe_pct_change(latest["delinquency_rate"], previous["delinquency_rate"])
        complaints_change_pct = _safe_pct_change(latest["complaints"], previous["complaints"])
        abnormal_events_change_pct = _safe_pct_change(latest["abnormal_events"], previous["abnormal_events"])
        real_estate_change_pct = _safe_pct_change(latest["exposure_real_estate"], previous["exposure_real_estate"])
        sme_change_pct = _safe_pct_change(latest["exposure_sme"], previous["exposure_sme"])

        if delinquency_change_pct >= 20:
            alerts.append({
                "company_name": company,
                "alert_type": "신용리스크 경보",
                "severity": "High",
                "detail": f"연체율이 전월 대비 {delinquency_change_pct}% 증가했습니다.",
                "recommended_action": "연체 증가 차주군과 포트폴리오를 우선 점검하고 취약 섹터를 재분석하세요.",
            })
        elif delinquency_change_pct <= -5:
            alerts.append({
                "company_name": company,
                "alert_type": "연체율 개선 포착",
                "severity": "Low",
                "detail": f"연체율이 전월 대비 {abs(delinquency_change_pct)}% 감소했습니다.",
                "recommended_action": "개선 요인이 지속 가능한지 확인하고 우수 사례를 타 포트폴리오에 확산하세요.",
            })

        if complaints_change_pct >= 18:
            alerts.append({
                "company_name": company,
                "alert_type": "민원 증가 경보",
                "severity": "Medium",
                "detail": f"민원 건수가 전월 대비 {complaints_change_pct}% 증가했습니다.",
                "recommended_action": "민원 유형을 세분화하고 반복 발생 프로세스를 우선 개선하세요.",
            })

        if abnormal_events_change_pct >= 20:
            alerts.append({
                "company_name": company,
                "alert_type": "운영리스크 경보",
                "severity": "High",
                "detail": f"이상 이벤트 수가 전월 대비 {abnormal_events_change_pct}% 증가했습니다.",
                "recommended_action": "이상 이벤트 발생 부서, 채널, 프로세스를 즉시 점검하세요.",
            })

        if real_estate_change_pct >= 15:
            alerts.append({
                "company_name": company,
                "alert_type": "부동산 익스포저 집중 경보",
                "severity": "Medium",
                "detail": f"부동산 익스포저가 전월 대비 {real_estate_change_pct}% 증가했습니다.",
                "recommended_action": "업종 집중도와 내부 한도 운영 상태를 재검토하세요.",
            })

        if sme_change_pct >= 12:
            alerts.append({
                "company_name": company,
                "alert_type": "SME 익스포저 확대 경보",
                "severity": "Medium",
                "detail": f"중소기업 익스포저가 전월 대비 {sme_change_pct}% 증가했습니다.",
                "recommended_action": "차주군별 건전성 변화를 점검하고 취약 업종 비중을 확인하세요.",
            })

    latest_month = metrics_df["date"].max().to_period("M")
    recent_logs = logs_df[logs_df["date"].dt.to_period("M") == latest_month].copy()
    if not recent_logs.empty:
        for _, row in recent_logs.iterrows():
            if row["severity"] in ["High", "Medium"]:
                alerts.append(
                    {
                        "company_name": row["company_name"],
                        "alert_type": row["issue_type"],
                        "severity": row["severity"],
                        "detail": row["description"],
                        "recommended_action": "세부 원인 로그를 검토하고 즉시 대응 계획을 수립하세요.",
                    }
                )

    if not alerts:
        return pd.DataFrame(columns=["company_name", "alert_type", "severity", "detail", "recommended_action"])

    alerts_df = pd.DataFrame(alerts).drop_duplicates()
    severity_rank = {"High": 3, "Medium": 2, "Low": 1}
    alerts_df["severity_rank"] = alerts_df["severity"].map(severity_rank)
    return alerts_df.sort_values(["severity_rank", "company_name"], ascending=[False, True]).drop(columns=["severity_rank"]).reset_index(drop=True)


def generate_executive_report(risk_df: pd.DataFrame, alerts_df: pd.DataFrame, latest_month: str) -> str:
    top_company = risk_df.sort_values("risk_score", ascending=False).iloc[0]
    top_alerts = alerts_df.head(4)

    lines = [
        f"[JB Insight CRO] {latest_month} 그룹 리스크 브리프",
        "",
        "1. 종합 요약",
        f"- 최고 위험 계열사는 {top_company['company_name']}이며 그룹 리스크 점수는 {int(top_company['risk_score'])}점입니다.",
        f"- 해당 계열사의 연체율은 전월 대비 {top_company['delinquency_change_pp']:+.2f}%p, 최근 3개월 평균 대비 {top_company['vs_3m_avg_pp']:+.2f}%p 변동했습니다.",
        f"- 경영진 관점 해석: {top_company['executive_headline']}",
        f"- 당월 식별된 주요 경보는 총 {len(alerts_df)}건입니다.",
        "",
        "2. 핵심 리스크 이슈",
    ]

    if len(top_alerts) == 0:
        lines.append("- 주요 경보 없음")
    else:
        for idx, (_, row) in enumerate(top_alerts.iterrows(), start=1):
            lines.append(f"- 이슈 {idx}: {row['company_name']} / {row['alert_type']} / {row['detail']}")

    lines += [
        "",
        "3. 권고 사항",
        "- 위험도 상위 계열사는 전월 대비 변화뿐 아니라 3개월 평균 대비 이탈 정도를 함께 관리해야 합니다.",
        "- 기업금융 포트폴리오 내 PF, 기업 운전자금, 담보부 대출 세그먼트를 분리 모니터링하고 차환·회수 이슈를 별도 관리해야 합니다.",
        "- 연체율 개선이 확인된 사례는 심사·회수 프로세스 관점에서 우수 사례로 전파합니다.",
    ]
    return "\n".join(lines)


def get_delinquency_snapshot(risk_df: pd.DataFrame, company_name: str) -> dict:
    row = risk_df[risk_df["company_name"] == company_name].iloc[0]
    direction = "개선" if row["delinquency_change_pp"] < 0 else "악화" if row["delinquency_change_pp"] > 0 else "유지"
    return {
        "current_rate": row["latest_delinquency_rate"],
        "previous_rate": row["previous_delinquency_rate"],
        "mom_change_pp": row["delinquency_change_pp"],
        "mom_change_pct": row["delinquency_change_pct"],
        "trailing_3m_avg": row["trailing_3m_avg"],
        "vs_3m_avg_pp": row["vs_3m_avg_pp"],
        "vs_3m_avg_pct": row["vs_3m_avg_pct"],
        "direction": direction,
        "headline": row["executive_headline"],
        "positive_driver_summary": row["positive_driver_summary"],
        "negative_driver_summary": row["negative_driver_summary"],
    }


def get_company_comparison(risk_df: pd.DataFrame) -> dict:
    worst = risk_df.sort_values(["delinquency_change_pp", "risk_score"], ascending=[False, False]).iloc[0]
    best = risk_df.sort_values(["delinquency_change_pp", "risk_score"], ascending=[True, False]).iloc[0]
    trend_table = risk_df[
        [
            "company_name",
            "company_type",
            "latest_delinquency_rate",
            "delinquency_change_pp",
            "vs_3m_avg_pp",
            "positive_driver_summary",
            "negative_driver_summary",
        ]
    ].rename(
        columns={
            "company_name": "계열사",
            "company_type": "유형",
            "latest_delinquency_rate": "현재 연체율",
            "delinquency_change_pp": "전월 대비 변화(%p)",
            "vs_3m_avg_pp": "3개월 평균 대비(%p)",
            "positive_driver_summary": "개선 요인",
            "negative_driver_summary": "악화 요인",
        }
    ).sort_values("전월 대비 변화(%p)", ascending=False)
    return {
        "best_company": best["company_name"],
        "best_change_pp": best["delinquency_change_pp"],
        "best_summary": best["positive_driver_summary"],
        "worst_company": worst["company_name"],
        "worst_change_pp": worst["delinquency_change_pp"],
        "worst_summary": worst["negative_driver_summary"],
        "trend_table": trend_table,
    }


def _merge_segment_periods(segment_df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    company_segments = segment_df[segment_df["company_name"] == company_name].copy()
    if company_segments.empty:
        return pd.DataFrame()

    # Backward-compatible schema normalization for older deployment data.
    text_defaults = {
        "portfolio_group": "기업금융",
        "collateral_type": "미분류 담보",
        "industry": "미분류 업종",
    }
    numeric_defaults = {
        "balance": 0,
        "delinquency_rate": 0.0,
        "customer_count": 0,
    }
    for col, default in text_defaults.items():
        if col not in company_segments.columns:
            company_segments[col] = default
        company_segments[col] = company_segments[col].fillna(default)
    for col, default in numeric_defaults.items():
        if col not in company_segments.columns:
            company_segments[col] = default
        company_segments[col] = company_segments[col].fillna(default)

    latest_period = company_segments["date"].max()
    previous_candidates = company_segments[company_segments["date"] < latest_period]["date"]
    if previous_candidates.empty:
        return pd.DataFrame()
    previous_period = previous_candidates.max()

    latest_seg = company_segments[company_segments["date"] == latest_period].rename(
        columns={
            "balance": "curr_balance",
            "delinquency_rate": "curr_delinquency_rate",
            "customer_count": "curr_customer_count",
        }
    )
    previous_seg = company_segments[company_segments["date"] == previous_period].rename(
        columns={
            "balance": "prev_balance",
            "delinquency_rate": "prev_delinquency_rate",
            "customer_count": "prev_customer_count",
        }
    )

    merge_keys = [col for col in ["company_name", "portfolio_group", "segment_name", "collateral_type", "industry"] if col in previous_seg.columns and col in latest_seg.columns]
    if not merge_keys:
        return pd.DataFrame()

    merged = previous_seg.merge(latest_seg, on=merge_keys, how="outer")
    for col in ["prev_balance", "curr_balance", "prev_delinquency_rate", "curr_delinquency_rate", "prev_customer_count", "curr_customer_count"]:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0)
    for col, default in text_defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)

    merged["delinquency_delta"] = (merged["curr_delinquency_rate"] - merged["prev_delinquency_rate"]).round(2)
    merged["balance_delta"] = (merged["curr_balance"] - merged["prev_balance"]).round(0)
    merged["customer_delta"] = (merged["curr_customer_count"] - merged["prev_customer_count"]).round(0)
    merged["latest_period"] = latest_period
    merged["previous_period"] = previous_period
    return merged


def get_enterprise_portfolio_summary(segment_df: pd.DataFrame, company_name: str) -> dict:
    merged = _merge_segment_periods(segment_df, company_name)
    if merged.empty:
        return {
            "worst_segment": "-",
            "worst_change_pp": 0.0,
            "best_segment": "-",
            "best_change_pp": 0.0,
            "pf_share": 0.0,
            "secured_share": 0.0,
            "largest_balance_segment": "-",
            "largest_balance": 0.0,
        }

    worst = merged.sort_values(["delinquency_delta", "curr_balance"], ascending=[False, False]).iloc[0]
    best = merged.sort_values(["delinquency_delta", "curr_balance"], ascending=[True, False]).iloc[0]
    total_balance = float(merged["curr_balance"].sum())
    pf_balance = float(merged[merged["segment_name"].str.contains("PF|프로젝트", regex=True)]["curr_balance"].sum())
    secured_balance = float(merged[merged["collateral_type"].str.contains("담보|토지|설비|재고|프로젝트", regex=True)]["curr_balance"].sum())
    largest = merged.sort_values("curr_balance", ascending=False).iloc[0]
    return {
        "worst_segment": worst["segment_name"],
        "worst_change_pp": worst["delinquency_delta"],
        "best_segment": best["segment_name"],
        "best_change_pp": best["delinquency_delta"],
        "pf_share": round((pf_balance / total_balance) * 100, 1) if total_balance else 0.0,
        "secured_share": round((secured_balance / total_balance) * 100, 1) if total_balance else 0.0,
        "largest_balance_segment": largest["segment_name"],
        "largest_balance": round(float(largest["curr_balance"]), 1),
    }


def generate_delinquency_reason_report(metrics_df: pd.DataFrame, drivers_df: pd.DataFrame, segment_df: pd.DataFrame, company_name: str) -> str:
    company_metrics = metrics_df[metrics_df["company_name"] == company_name].sort_values("date")
    if len(company_metrics) < 2:
        return f"[{company_name}] 연체율 변동 분석\n\n비교 가능한 기간 데이터가 부족합니다."

    latest, previous = _latest_and_previous(company_metrics)
    current_rate = latest["delinquency_rate"]
    previous_rate = previous["delinquency_rate"]
    change_pp = _delta_pp(current_rate, previous_rate)
    trailing_3m_avg = _previous_n_average(company_metrics, "delinquency_rate", 3)
    vs_3m_avg_pp = _delta_pp(current_rate, trailing_3m_avg)

    company_drivers = drivers_df[drivers_df["company_name"] == company_name].copy()
    company_drivers["abs_contribution"] = company_drivers["contribution_bps"].abs()
    company_drivers = company_drivers.sort_values("abs_contribution", ascending=False)
    positive_summary = _build_driver_summary(company_drivers, direction="positive")
    negative_summary = _build_driver_summary(company_drivers, direction="negative")

    seg_merged = _merge_segment_periods(segment_df, company_name)
    if seg_merged.empty:
        return f"[{company_name}] 연체율 변동 분석\n\n세그먼트 데이터가 부족합니다."

    worsening_segments = seg_merged.sort_values(["delinquency_delta", "curr_balance"], ascending=[False, False]).head(3)
    improving_segments = seg_merged.sort_values(["delinquency_delta", "curr_balance"], ascending=[True, False]).head(2)
    portfolio_summary = get_enterprise_portfolio_summary(segment_df, company_name)

    lines = [
        f"[{company_name}] 기업금융 연체율 원인 분석 보고",
        "",
        "1. Executive Summary",
        f"- 기준 기간: {previous['date'].strftime('%Y-%m')} → {latest['date'].strftime('%Y-%m')}",
        f"- 전사 연체율은 {previous_rate:.2f}%에서 {current_rate:.2f}%로 {change_pp:+.2f}%p 변동했습니다.",
        f"- 최근 3개월 평균({trailing_3m_avg:.2f}%) 대비 현재 수준은 {vs_3m_avg_pp:+.2f}%p입니다.",
        f"- 기업금융 포트폴리오 내 PF/프로젝트금융 비중은 {portfolio_summary['pf_share']}%, 담보 기반 익스포저 비중은 {portfolio_summary['secured_share']}%입니다.",
    ]

    if change_pp > 0:
        lines.append(f"- 종합 판단: 연체율 상승은 {negative_summary} 중심으로 발생했으며, 특히 {portfolio_summary['worst_segment']} 세그먼트의 악화가 전체 포트폴리오 부담을 확대했습니다.")
    elif change_pp < 0:
        lines.append(f"- 종합 판단: 연체율 하락은 {positive_summary} 중심의 개선 효과가 반영된 결과이며, {portfolio_summary['best_segment']} 세그먼트의 안정화가 기여했습니다.")
    else:
        lines.append("- 종합 판단: 전사 연체율 수준은 유사하나 세그먼트별 편차가 존재하므로 PF와 담보대출 중심의 미시 점검이 필요합니다.")

    lines += [
        "",
        "2. 주요 원인 진단",
    ]
    if company_drivers.empty:
        lines.append("- 식별된 드라이버 데이터가 없습니다.")
    else:
        for _, row in company_drivers.head(5).iterrows():
            impact = "개선" if row["direction"] == "positive" else "악화"
            lines.append(f"- {row['driver_name']} · {impact} 기여 {abs(int(row['contribution_bps']))}bp · {row['description']}")

    lines += [
        "",
        "3. 기업금융 세그먼트 세부 분석",
    ]
    for _, row in worsening_segments.iterrows():
        lines.append(
            f"- 악화 세그먼트: {row['segment_name']} / 담보유형 {row['collateral_type']} / 업종 {row['industry']} · 연체율 {row['prev_delinquency_rate']:.2f}% → {row['curr_delinquency_rate']:.2f}% (Δ {row['delinquency_delta']:+.2f}%p), 잔액 {row['prev_balance']:.0f} → {row['curr_balance']:.0f}"
        )
    for _, row in improving_segments.iterrows():
        if row['delinquency_delta'] < 0:
            lines.append(
                f"- 개선 세그먼트: {row['segment_name']} / 담보유형 {row['collateral_type']} / 업종 {row['industry']} · 연체율 {row['prev_delinquency_rate']:.2f}% → {row['curr_delinquency_rate']:.2f}% (Δ {row['delinquency_delta']:+.2f}%p), 잔액 {row['prev_balance']:.0f} → {row['curr_balance']:.0f}"
            )

    lines += [
        "",
        "4. Management Implication",
        f"- 우선 점검 대상은 {portfolio_summary['worst_segment']}이며, 차환일정·담보재평가·회수전략을 패키지로 점검할 필요가 있습니다.",
        f"- 방어력 확보 사례는 {portfolio_summary['best_segment']}에서 확인되며, 심사 기준과 조기회수 정책을 타 세그먼트에 확산할 수 있습니다.",
        f"- 잔액 기준 최대 포트폴리오는 {portfolio_summary['largest_balance_segment']}({portfolio_summary['largest_balance']})로, 자산 규모와 건전성 변화를 함께 관리해야 합니다.",
        "- 보고 체계는 전월 대비 변화와 3개월 평균 대비 이탈을 병행 관리해 단기 이상징후와 구조적 추세를 동시에 추적해야 합니다.",
    ]
    return "\n".join(lines)


def get_segment_detail_table(segment_df: pd.DataFrame, company_name: str) -> pd.DataFrame:
    merged = _merge_segment_periods(segment_df, company_name)
    if merged.empty:
        return pd.DataFrame()
    result = merged.rename(
        columns={
            "portfolio_group": "포트폴리오군",
            "segment_name": "세그먼트",
            "collateral_type": "담보유형",
            "industry": "업종",
            "prev_delinquency_rate": "전월 연체율",
            "curr_delinquency_rate": "현재 연체율",
            "prev_balance": "전월 잔액",
            "curr_balance": "현재 잔액",
            "prev_customer_count": "전월 차주수",
            "curr_customer_count": "현재 차주수",
        }
    )
    result["연체율 변화(%p)"] = result["delinquency_delta"]
    result["잔액 변화"] = result["balance_delta"]
    result["차주수 변화"] = result["customer_delta"]
    return result[
        [
            "포트폴리오군",
            "세그먼트",
            "담보유형",
            "업종",
            "전월 연체율",
            "현재 연체율",
            "연체율 변화(%p)",
            "전월 잔액",
            "현재 잔액",
            "잔액 변화",
            "전월 차주수",
            "현재 차주수",
            "차주수 변화",
        ]
    ].sort_values(["연체율 변화(%p)", "현재 잔액"], ascending=[False, False])


def answer_question(question: str, risk_df: pd.DataFrame, alerts_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str:
    top_company = risk_df.sort_values("risk_score", ascending=False).iloc[0]

    if "가장 위험한 계열사" in question:
        return f"이번 달 가장 위험한 계열사는 {top_company['company_name']}입니다. 리스크 점수는 {int(top_company['risk_score'])}점이며 주요 위험 요인은 {top_company['top_drivers_text']}입니다."

    if "가장 크게 악화된 지표" in question:
        metric_map = {
            "연체율": top_company["delinquency_change_pct"],
            "민원 건수": top_company["complaints_change_pct"],
            "이상 이벤트 수": top_company["abnormal_events_change_pct"],
            "부동산 익스포저": top_company["real_estate_change_pct"],
            "SME 익스포저": top_company["sme_change_pct"],
        }
        metric_name = max(metric_map, key=metric_map.get)
        return f"지난달 대비 가장 크게 악화된 지표는 {top_company['company_name']}의 {metric_name}입니다. 변화율은 {metric_map[metric_name]}%입니다."

    if "우선 대응" in question:
        if len(alerts_df) == 0:
            return "현재 우선 대응이 필요한 경보가 없습니다."
        top_alert = alerts_df.iloc[0]
        return f"우선 대응이 필요한 리스크는 {top_alert['company_name']}의 {top_alert['alert_type']}입니다. 사유는 '{top_alert['detail']}'이며 권고 조치는 '{top_alert['recommended_action']}'입니다."

    if "운영리스크" in question:
        ranked = risk_df.sort_values(["operational_score", "risk_score"], ascending=[False, False]).iloc[0]
        return f"운영리스크가 가장 큰 계열사는 {ranked['company_name']}입니다. 운영리스크 점수는 {int(ranked['operational_score'])}점이며 전체 리스크 점수는 {int(ranked['risk_score'])}점입니다."

    return "지원하지 않는 질문입니다."
