from __future__ import annotations

import json
from typing import Any

from loguru import logger
from openai import OpenAI

from wadipulse.config import Settings
from wadipulse.storage.duckdb_store import WadiPulseStore

AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_region_water_budget",
            "description": (
                "Retrieve aggregated irrigated area (hectares), mean NDVI, mean NDMI moisture index, "
                "and count of water-stressed center-pivot fields by agricultural basin."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Optional basin key: 'wadi_ad_dawasir' or 'al_kharj'. Omit for all basins.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_high_stress_pivots",
            "description": (
                "List individual center-pivot fields experiencing severe or moderate canopy moisture stress "
                "ranked by stress_score and area_ha."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Optional basin key: 'wadi_ad_dawasir' or 'al_kharj'.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["severe_stress", "moderate_stress", "healthy"],
                        "description": "Filter by biophysical stress classification.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of fields to return (default 10).",
                    },
                },
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are WadiPulse GeoAgent, a hydrology and precision agriculture analyst specialized in "
    "arid center-pivot irrigation across Saudi Arabia (Wadi ad-Dawasir and Al-Kharj). "
    "Always ground your answers in empirical numbers retrieved from the DuckDB biophysical tools "
    "(hectares, NDVI, NDMI, field IDs, coordinates). Be concise, scientific, and actionable."
)


class WadiPulseGeoAgent:
    """Tool-calling Geospatial LLM Agent grounded in DuckDB pivot field telemetry."""

    def __init__(
            self,
            store: WadiPulseStore | None = None,
            settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.store = store or WadiPulseStore(self.settings.db_path)

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        if name == "query_region_water_budget":
            data = self.store.get_region_summary(region=args.get("region"))
            return json.dumps(data, ensure_ascii=False)
        if name == "find_high_stress_pivots":
            data = self.store.get_fields(
                region=args.get("region"),
                status=args.get("status"),
                limit=int(args.get("limit", 10)),
            )
            return json.dumps(data, ensure_ascii=False)
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _deterministic_synthesis(self, question: str) -> dict[str, Any]:
        """Grounded analytical synthesis directly over DuckDB SQL telemetry."""
        q_lower = question.lower()
        is_comparison = any(w in q_lower for w in ("compare", "between", "vs", "versus", "difference"))

        target_region: str | None = None
        if not is_comparison:
            if "wadi" in q_lower or "dawasir" in q_lower:
                target_region = "wadi_ad_dawasir"
            elif "kharj" in q_lower:
                target_region = "al_kharj"

        summaries = self.store.get_region_summary(region=target_region)
        stressed = self.store.get_fields(region=target_region, status="severe_stress", limit=5)
        if not stressed:
            stressed = self.store.get_fields(region=target_region, status="moderate_stress", limit=5)

        if not summaries:
            return {
                "answer": (
                    "The DuckDB telemetry store is currently empty. Run a region analysis first "
                    "to segment center-pivot fields with Prithvi-EO-2.0 and populate biophysical metrics."
                ),
                "tool_calls": ["query_region_water_budget"],
                "mode": "deterministic_duckdb_agent",
            }

        lines: list[str] = []

        if any(w in q_lower for w in ("what is", "about", "explain", "overview")):
            lines.append(
                "**Basin Hydrological Context:** **Wadi ad-Dawasir** (south-central Saudi Arabia) and "
                "**Al-Kharj** (south-east of Riyadh) are major hyper-arid agricultural basins relying on "
                "center-pivot groundwater irrigation from fossil aquifers.\n"
            )

        if is_comparison and len(summaries) >= 2:
            by_reg = {s["region"]: s for s in summaries}
            w = by_reg.get("wadi_ad_dawasir")
            k = by_reg.get("al_kharj")
            if w and k:
                w_severe_pct = round(100.0 * w["severe_count"] / max(w["total_pivots"], 1), 1)
                k_severe_pct = round(100.0 * k["severe_count"] / max(k["total_pivots"], 1), 1)
                area_ratio = round(w["total_irrigated_ha"] / max(k["total_irrigated_ha"], 1e-6), 1)
                lines.append(
                    f"**Cross-Basin Comparison (`Wadi ad-Dawasir` vs. `Al-Kharj`):**\n"
                    f"- **Irrigated Footprint:** Wadi ad-Dawasir covers **{w['total_irrigated_ha']} ha** "
                    f"across **{w['total_pivots']}** active pivots (~**{area_ratio}×** larger than Al-Kharj's "
                    f"**{k['total_irrigated_ha']} ha** across **{k['total_pivots']}** pivots).\n"
                    f"- **Canopy Moisture Stress:** Al-Kharj exhibits a higher proportion of severe stress "
                    f"(**{k_severe_pct}%** of pivots, mean NDMI `{k['avg_ndmi']}`) compared to "
                    f"Wadi ad-Dawasir (**{w_severe_pct}%** severe stress, mean NDMI `{w['avg_ndmi']}`).\n"
                )

        lines.append("**Regional Irrigated Water-Stress Synthesis (Grounded in DuckDB Telemetry):**")
        for s in summaries:
            lines.append(
                f"- **{s['region']}** ({s['acquisition_date']}): **{s['total_pivots']}** active pivots "
                f"covering **{s['total_irrigated_ha']} ha** | Mean NDVI: `{s['avg_ndvi']}`, "
                f"Mean NDMI: `{s['avg_ndmi']}` | Severe stress: `{int(s['severe_count'])}`, "
                f"Moderate stress: `{int(s['moderate_count'])}`, Healthy: `{int(s['healthy_count'])}`."
            )

        if stressed:
            lines.append("\n**Priority Fields Requiring Irrigation Audit:**")
            for f in stressed[:5]:
                lines.append(
                    f"- `{f['field_id']}` ({f['region']}, **{f['area_ha']} ha**, "
                    f"lat `{f['centroid_lat']}`, lon `{f['centroid_lon']}`): "
                    f"NDMI=`{f['mean_ndmi']}`, NDVI=`{f['mean_ndvi']}`, "
                    f"stress_score=`{f['stress_score']}` ({f['stress_status']})."
                )

        return {
            "answer": "\n".join(lines),
            "tool_calls": ["query_region_water_budget", "find_high_stress_pivots"],
            "mode": "deterministic_duckdb_agent",
        }

    def ask(self, question: str) -> dict[str, Any]:
        try:
            client = OpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=5.0,
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            executed_tools: list[str] = []

            first_resp = client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = first_resp.choices[0].message
            if msg.tool_calls:
                messages.append(msg.model_dump(exclude_none=True))
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or "{}")
                    executed_tools.append(fn_name)
                    tool_output = self._dispatch_tool(fn_name, fn_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_output,
                        }
                    )

                second_resp = client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=messages,
                    temperature=0.2,
                )
                return {
                    "answer": second_resp.choices[0].message.content or "",
                    "tool_calls": executed_tools,
                    "mode": f"llm:{self.settings.llm_model}",
                }

            return {
                "answer": msg.content or "",
                "tool_calls": executed_tools,
                "mode": f"llm:{self.settings.llm_model}",
            }
        except Exception as exc:
            logger.info(f"LLM endpoint unavailable ({exc}); using deterministic DuckDB synthesis")
            return self._deterministic_synthesis(question)
