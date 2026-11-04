"""Streamlit decision-support app for the transfer window (participant 2)."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_data import (
    FINANCIAL_DISCLAIMER,
    SUPPORTED_POSITIONS,
    load_players,
    load_role_requirements,
    metric_labels,
    role_weights,
)
from src.explain import candidate_funnel, pool_quantiles, positions_overview, score_gap, weight_table
from src.ranking import filter_candidates, score_players
from src.scenarios import (
    CORE_SIZE,
    PROSPECT_AGE,
    PROSPECT_MINUTES,
    VETERAN_AGE,
    budget_sensitivity,
    plan_steps,
    reinforcement_plan,
    squad_audit,
    team_core_quality,
)

DEFAULT_TEAM = "Manchester United"
MILLION = 1e6
FLAG_LABELS = {
    "": "",
    "veteran_below_median": "ветеран ниже медианы: продать или продлить?",
    "prospect_low_minutes": "молодой без практики: аренда?",
}
DECISION_LABELS = {
    "signed": "куплен",
    "too_expensive": "дороже остатка — пропуск",
    "no_value": "стоимость неизвестна — на оценку",
    "slots_full": "слоты заняты",
}
BASE_COLUMNS = ["player_name", "team", "league", "position", "age", "minutes"]
COLUMN_CONFIG = {
    "player_name": "Игрок",
    "team": "Клуб",
    "league": "Лига",
    "position": "Поз.",
    "age": "Возраст",
    "minutes": "Минуты",
    "score": st.column_config.ProgressColumn("Рейтинг роли", min_value=0, max_value=100, format="%.0f"),
    "baseline_score": st.column_config.ProgressColumn(
        "База: голы+ассисты", min_value=0, max_value=100, format="%.0f"
    ),
    "hidden_gem": st.column_config.CheckboxColumn("Скрытый талант"),
    "market_value": st.column_config.NumberColumn("Стоимость EA, €M", format="%.1f"),
    "salary": st.column_config.NumberColumn("Зарплата EA, €/нед.", format="%d"),
    "value_for_money": st.column_config.NumberColumn("Рейтинг на €M", format="%.1f"),
    "contract_end": st.column_config.NumberColumn("Контракт до", format="%d"),
    "n_missing_metrics": "Нет метрик",
    "flag": "Сигнал",
}

st.set_page_config(page_title="Трансферный аудит", layout="wide")


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_players(), load_role_requirements()


try:
    players, roles = load_data()
except FileNotFoundError as error:
    st.error(f"{error}. Сначала подготовьте данные: `python -m src.prepare_top5`.")
    st.stop()
labels = metric_labels(roles)


def show_players(frame: pd.DataFrame, extra_columns: list[str]) -> None:
    """Render a player table with score bars, money in millions and readable metric names."""
    columns = BASE_COLUMNS + [column for column in extra_columns if column in frame]
    table = frame[columns].copy()
    if "market_value" in table:
        table["market_value"] = table["market_value"] / MILLION
    st.dataframe(table, hide_index=True, column_config={**labels, **COLUMN_CONFIG}, width="stretch")


def player_options(frame: pd.DataFrame) -> dict[str, str]:
    """Map a readable label to record_id; the team disambiguates namesakes."""
    return {f"{row.player_name} ({row.team})": row.record_id for row in frame.itertuples()}


def names_with_scores(frame: pd.DataFrame) -> str:
    """'Name (72.9), Name (67.5)' for short textual explanations."""
    return ", ".join(f"{name} ({score:.1f})" for name, score in zip(frame["player_name"], frame["score"]))


# ---------------------------------------------------------------- inputs
with st.sidebar:
    st.header("Параметры")
    teams = sorted(players["team"].unique())
    team = st.selectbox("Клуб", teams, index=teams.index(DEFAULT_TEAM) if DEFAULT_TEAM in teams else 0)
    position = st.radio("Позиция", SUPPORTED_POSITIONS, index=1, horizontal=True)
    min_minutes = st.slider(
        "Минимум минут за сезон", 0, 2700, 900, step=90,
        help="Показатели на 90 минут нестабильны при малой выборке; 900 минут ≈ 10 полных матчей.",
    )
    st.subheader("Фильтры кандидатов")
    age_range = st.slider("Возраст", 16, 40, (18, 29))
    max_value = st.slider("Макс. стоимость, €M", 0, 200, 60, step=5)
    include_unknown = st.checkbox("Показывать игроков с неизвестной стоимостью", value=True)
    all_leagues = sorted(players["league"].unique())
    leagues = st.multiselect("Лиги", all_leagues, default=all_leagues)
    st.subheader("Сценарий усиления")
    budget = st.slider("Бюджет, €M", 0, 300, 80, step=5)
    n_signings = st.slider("Количество покупок", 1, 3, 1)
    with st.expander("Веса роли (тактическая модель)"):
        weights = {
            metric: st.slider(labels[metric], 0.0, 1.0, weight, 0.05, key=f"{position}:{metric}")
            for metric, weight in role_weights(roles, position).items()
        }

try:
    scored = score_players(players, position, weights, min_minutes)
except ValueError as error:
    st.warning(str(error))
    st.stop()
squad = scored.loc[scored["team"].eq(team)]
candidates = filter_candidates(
    scored,
    exclude_team=team,
    age_range=age_range,
    max_value=max_value * MILLION,
    include_unknown_value=include_unknown,
    leagues=leagues,
)

st.title(f"Трансферный аудит: {team}, позиция {position}")
st.caption(
    f"Сезон 2025/26, пять топ-лиг. В пуле позиции {len(scored)} игроков с ≥ {min_minutes} минут; "
    f"рейтинг = взвешенные процентили внутри пула (100 = лучший). {FINANCIAL_DISCLAIMER}"
)
tab_audit, tab_candidates, tab_compare, tab_plan, tab_calc = st.tabs(
    ["Аудит состава", "Кандидаты", "Сравнение", "Сценарий усиления", "Вычисления"]
)

# ---------------------------------------------------------------- squad audit
with tab_audit:
    audit = squad_audit(players, scored, team, position)
    league_table = team_core_quality(scored)
    rank = league_table.index.get_loc(team) + 1 if team in league_table.index else None
    left, middle, right = st.columns(3)
    left.metric("Игроков клуба на позиции", len(audit))
    middle.metric("Из них с рейтингом", int(audit["score"].notna().sum()))
    right.metric(
        "Качество ядра (средний рейтинг топ-3)",
        f"{league_table.get(team, float('nan')):.0f} / 100",
        f"{rank}-е место из {len(league_table)} клубов" if rank else None,
        delta_color="off",
    )
    show_players(
        audit.assign(flag=audit["flag"].map(FLAG_LABELS)),
        ["score", "baseline_score", "hidden_gem", "market_value", "contract_end", "flag"],
    )
    st.caption(
        "Игроки ниже порога минут остаются в таблице без рейтинга. "
        "«Сигнал» — эвристика для обсуждения, а не решение."
    )

# ---------------------------------------------------------------- candidates
with tab_candidates:
    top_n = st.slider("Показать кандидатов", 5, 50, 15, step=5)
    show_players(
        candidates.head(top_n),
        ["score", "baseline_score", "hidden_gem", "market_value", "value_for_money", "n_missing_metrics"],
    )
    st.caption(
        "«Скрытый талант» — топ-25% по модели роли, но ниже медианы по простому рейтингу голы+ассисты. "
        "«Нет метрик» — сколько метрик отсутствует у игрока (учтены нейтрально, как 50-й процентиль)."
    )
    if not candidates.empty:
        st.subheader("Из чего складывается рейтинг")
        options = player_options(candidates.head(top_n))
        chosen = options[st.selectbox("Игрок", list(options))]
        player = candidates.loc[candidates["record_id"].eq(chosen)].iloc[0]
        breakdown = pd.DataFrame(
            {
                "Метрика": [labels[metric] for metric in weights],
                "Значение": [player[metric] for metric in weights],
                "Процентиль": [100 * player[f"pct_{metric}"] for metric in weights],
                "Баллы": [player[f"contrib_{metric}"] for metric in weights],
            }
        ).set_index("Метрика").sort_values("Баллы", ascending=False)
        bars = go.Figure(go.Bar(x=breakdown["Баллы"], y=breakdown.index, orientation="h"))
        bars.update_layout(yaxis={"autorange": "reversed"}, height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        chart, table = st.columns([2, 1])
        chart.plotly_chart(bars, width="stretch")
        table.dataframe(breakdown.round(2), width="stretch")

# ---------------------------------------------------------------- comparison
with tab_compare:
    options = player_options(pd.concat([squad, candidates.head(50)]))
    default = list(player_options(squad.head(2))) + list(player_options(candidates.head(1)))
    selected = st.multiselect("Игроки для сравнения (2–4)", list(options), default=default, max_selections=4)
    if len(selected) < 2:
        st.info("Выберите хотя бы двух игроков.")
    else:
        compared = pd.concat([squad, candidates]).set_index("record_id").loc[[options[label] for label in selected]]
        radar = go.Figure()
        for label, (_, player) in zip(selected, compared.iterrows()):
            radar.add_trace(
                go.Scatterpolar(
                    r=[100 * player[f"pct_{metric}"] for metric in weights],
                    theta=[labels[metric] for metric in weights],
                    fill="toself",
                    name=label,
                )
            )
        radar.update_layout(polar={"radialaxis": {"range": [0, 100]}}, height=520)
        st.plotly_chart(radar, width="stretch")
        st.caption("Оси радара — процентили внутри пула позиции; в таблице — исходные значения метрик.")
        show_players(compared, list(weights) + ["score", "market_value"])

# ---------------------------------------------------------------- reinforcement plan
with tab_plan:
    plan = reinforcement_plan(candidates, squad, budget * MILLION, n_signings)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Потрачено, €M", f"{plan.spent / MILLION:.1f}")
    c2.metric("Остаток, €M", f"{plan.remaining / MILLION:.1f}")
    c3.metric("Качество ядра до", f"{plan.quality_before:.0f}")
    c4.metric("Качество ядра после", f"{plan.quality_after:.0f}", f"{plan.quality_after - plan.quality_before:+.1f}")
    if plan.signings.empty:
        st.warning("В рамках бюджета и фильтров нет кандидатов с известной стоимостью.")
    else:
        show_players(plan.signings, ["score", "hidden_gem", "market_value", "salary", "contract_end"])
    if not plan.needs_valuation.empty:
        st.info("Сильные кандидаты без оценки стоимости: нужна оценка скаутов, автоматически не покупаются.")
        show_players(plan.needs_valuation, ["score", "hidden_gem"])

    st.subheader("Что изменится при другом бюджете")
    sensitivity = budget_sensitivity(candidates, squad, budget * MILLION, n_signings=n_signings)
    st.dataframe(
        pd.DataFrame(
            {
                "Бюджет": sensitivity["budget_change"].map("{:+.0%}".format),
                "Бюджет, €M": sensitivity["budget"] / MILLION,
                "Покупки": sensitivity["signings"],
                "Потрачено, €M": sensitivity["spent"] / MILLION,
                "Качество ядра после": sensitivity["quality_after"].round(1),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Жадный алгоритм: берём лучшего по рейтингу кандидата, который помещается в остаток бюджета, "
        "пока не кончатся бюджет или слоты. Качество ядра = средний рейтинг трёх лучших игроков позиции."
    )

# ---------------------------------------------------------------- calculations
with tab_calc:
    st.markdown(
        "Слева — метрики и веса (тактическая модель), справа — результаты. "
        "Здесь показано, как из весов и данных получаются числа на каждой вкладке."
    )

    st.subheader("1. Формула рейтинга и текущие веса")
    st.latex(r"score = \sum_i 100 \cdot \frac{w_i}{\sum_j w_j} \cdot pct_i")
    weight_summary = weight_table(weights)
    st.dataframe(
        weight_summary.assign(metric=weight_summary["metric"].map(labels)).round(3),
        hide_index=True,
        column_config={"metric": "Метрика", "weight": "Вес (слайдер)", "share": "Доля веса", "max_points": "Макс. баллов"},
        width="stretch",
    )
    st.caption(
        f"Пул: {len(scored)} игроков позиции {position} с ≥ {min_minutes} минут. Процентиль игрока — доля пула "
        "с меньшим значением метрики. Что означают процентили в единицах метрик:"
    )
    st.dataframe(
        pool_quantiles(scored, list(weights)).rename(index=labels).round(2),
        column_config={"p25": "25-й процентиль", "p50": "медиана", "p75": "75-й", "p90": "90-й"},
        width="stretch",
    )

    st.subheader("2. Аудит состава: качество ядра и место клуба")
    if squad.empty:
        st.info("У клуба нет игроков этой позиции в пуле.")
    else:
        core = squad.head(CORE_SIZE)
        st.markdown(
            f"Ядро = {CORE_SIZE} лучших игрока клуба по рейтингу: {names_with_scores(core)} "
            f"→ среднее **{core['score'].mean():.1f}**. То же считается для каждого клуба пула:"
        )
        standings = league_table.rename("core_quality").reset_index()
        standings.index = standings.index + 1
        shown_rows = standings.head(5) if rank is None else pd.concat([standings.head(5), standings.loc[[rank]]])
        st.dataframe(
            shown_rows.drop_duplicates(),
            column_config={"team": "Клуб", "core_quality": st.column_config.NumberColumn("Качество ядра", format="%.1f")},
            width="stretch",
        )
    overview = positions_overview(players, roles, team, min_minutes, {position: weights})
    st.dataframe(
        overview,
        hide_index=True,
        column_config={
            "position": "Позиция",
            "core_quality": st.column_config.NumberColumn("Качество ядра", format="%.1f"),
            "rank": "Место",
            "teams": "Клубов в пуле",
        },
        width="stretch",
    )
    if overview["rank"].notna().any():
        weakest = overview.loc[overview["rank"].idxmax()]
        st.markdown(
            f"Вывод: относительно других клубов слабее всего позиция **{weakest['position']}** "
            f"({int(weakest['rank'])}-е место из {int(weakest['teams'])}). Для текущей позиции взяты веса "
            "со слайдеров, для остальных — значения по умолчанию из `role_requirements.csv`."
        )
    st.markdown(
        f"Сигналы в аудите: медиана рейтинга пула = {scored['score'].median():.1f}; «ветеран ниже медианы» = "
        f"возраст ≥ {VETERAN_AGE} и рейтинг ниже медианы; «молодой без практики» = возраст ≤ {PROSPECT_AGE} "
        f"и меньше {PROSPECT_MINUTES} минут."
    )

    st.subheader("3. Кандидаты: воронка фильтров и разложение рейтинга")
    funnel = candidate_funnel(scored, team, age_range, leagues, max_value * MILLION, include_unknown)
    funnel_labels = {
        "pool": f"Пул позиции (≥ {min_minutes} минут)",
        "one_stint": "Один отрезок сезона на игрока (переходы зимой)",
        "exclude_team": f"Без игроков {team}",
        "age": f"Возраст {age_range[0]}–{age_range[1]}",
        "leagues": f"Лиги: {len(leagues)} из {len(all_leagues)}",
        "value": f"Стоимость ≤ {max_value} €M, неизвестная {'включена' if include_unknown else 'исключена'}",
    }
    st.dataframe(
        funnel.assign(step=funnel["step"].map(funnel_labels)),
        hide_index=True,
        column_config={"step": "Фильтр (накопительно)", "players": "Осталось игроков"},
        width="stretch",
    )
    if not candidates.empty:
        top = candidates.head(10)
        stack = go.Figure(
            [go.Bar(name=labels[metric], x=top["player_name"], y=top[f"contrib_{metric}"]) for metric in weights]
        )
        stack.update_layout(
            barmode="stack", height=400, yaxis_title="Баллы",
            legend={"orientation": "h", "y": -0.35}, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(stack, width="stretch")
        leader = top.iloc[0]
        contributions = pd.Series({labels[m]: leader[f"contrib_{m}"] for m in weights}).sort_values(ascending=False)
        strongest = ", ".join(f"{metric} ({points:.1f})" for metric, points in contributions.head(2).items())
        st.markdown(
            f"Лидер списка — **{leader['player_name']}** ({leader['score']:.1f} балла): больше всего дают "
            f"{strongest}. Столбцы выше — те же слагаемые для первых {len(top)} кандидатов."
        )
        st.markdown(
            f"«Скрытый талант» = рейтинг не ниже ≈{scored['score'].quantile(0.75):.1f} (граница верхней четверти "
            f"пула) и база голы+ассисты < 50. Среди первых {len(top)} кандидатов таких {int(top['hidden_gem'].sum())}."
        )

    st.subheader("4. Сравнение: откуда берётся разница в рейтинге")
    if len(selected) < 2:
        st.info("Выберите игроков на вкладке «Сравнение».")
    else:
        gap = score_gap(compared, weights, selected).rename(index=labels).rename_axis("Метрика")
        gap.columns = [
            column.replace("max_points", "Макс. баллов").replace("points:", "Баллы: ").replace("gap:", "Δ к первому: ")
            for column in gap.columns
        ]
        st.dataframe(gap.round(1), width="stretch")
        st.caption("Баллы по метрике = макс. баллов × процентиль игрока; Δ — разница с первым выбранным игроком.")
        for other in selected[1:]:
            delta = gap[f"Δ к первому: {other}"]
            st.markdown(
                f"**{other}** против **{selected[0]}**: разница рейтинга {delta.sum():+.1f}; больше всего "
                f"добавляет {delta.idxmax()} ({delta.max():+.1f}), больше всего отнимает {delta.idxmin()} "
                f"({delta.min():+.1f})."
            )

    st.subheader("5. Сценарий усиления: журнал жадного выбора")
    steps = plan.steps
    signed_rows = [row for row, decision in enumerate(steps["decision"]) if decision == "signed"]
    shown_steps = steps.head(signed_rows[-1] + 4 if signed_rows else 10)
    st.dataframe(
        shown_steps.assign(
            decision=shown_steps["decision"].map(DECISION_LABELS),
            market_value=shown_steps["market_value"] / MILLION,
            remaining_before=shown_steps["remaining_before"] / MILLION,
            remaining_after=shown_steps["remaining_after"] / MILLION,
        ),
        hide_index=True,
        column_config={
            "player_name": "Игрок",
            "team": "Клуб",
            "score": COLUMN_CONFIG["score"],
            "market_value": st.column_config.NumberColumn("Стоимость, €M", format="%.1f"),
            "remaining_before": st.column_config.NumberColumn("Остаток до, €M", format="%.1f"),
            "decision": "Решение",
            "remaining_after": st.column_config.NumberColumn("Остаток после, €M", format="%.1f"),
        },
        width="stretch",
    )
    after_core = pd.concat([squad, plan.signings]).nlargest(CORE_SIZE, "score")
    st.markdown(
        f"Ядро до: {names_with_scores(squad.head(CORE_SIZE)) or '—'} → **{plan.quality_before:.1f}**. "
        f"Ядро после: {names_with_scores(after_core) or '—'} → **{plan.quality_after:.1f}**."
    )
    cut_steps = plan_steps(candidates, 0.8 * budget * MILLION, n_signings)
    cut_signings = cut_steps.loc[cut_steps["decision"].eq("signed"), "player_name"].tolist()
    same_choice = cut_signings == plan.signings["player_name"].tolist()
    st.markdown(
        f"Тот же проход при бюджете −20 % ({0.8 * budget:.0f} €M) даёт: {', '.join(cut_signings) or 'ни одной покупки'}. "
        + (
            "Выбор не меняется: лучшие кандидаты стоят меньше уменьшенного бюджета."
            if same_choice
            else "Выбор меняется: кандидат с большим рейтингом перестаёт помещаться в остаток."
        )
    )

with st.expander("Ограничения и допущения"):
    st.markdown(
        """
- **Стоимость и зарплата** — оценки EA FC 26 без даты снимка, не реальный трансферный рынок; у ~15% игроков их нет.
- **Разные лиги** смешаны в одном пуле: процентили не учитывают уровень чемпионата и стиль команды
  (например, выносы и отборы выше у обороняющихся команд).
- **Малое игровое время** раздувает показатели на 90 минут; порог минут регулируется, по умолчанию 900.
- **Веса роли** — экспертное допущение (тактическая модель), а не результат обучения; их можно менять вживую.
- Рейтинг показывает соответствие ролевым требованиям, а не гарантию успеха трансфера.
- Вратари не поддерживаются: в таблице нет вратарских метрик.
        """
    )
