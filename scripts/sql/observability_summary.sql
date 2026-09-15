\x on
\pset pager off

WITH gate_summary AS (
    SELECT
        request_id,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE gate_name IN ('scope', 'scope_gate')
              AND phase = 'input'
        ), 0) AS scope_ms,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE gate_name IN ('censorship', 'censorship_gate')
              AND phase = 'input'
        ), 0) AS input_censorship_ms,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE phase = 'output'
               OR gate_name IN ('censorship_output', 'output_censorship')
        ), 0) AS output_censorship_ms,
        COUNT(*) FILTER (WHERE success = false) AS failed_gate_calls,
        STRING_AGG(DISTINCT error_type, ', ')
            FILTER (WHERE error_type IS NOT NULL) AS gate_errors
    FROM gate_check_log
    GROUP BY request_id
),
llm_summary AS (
    SELECT
        rt.request_id,
        COUNT(lcm.id) AS llm_calls,
        COALESCE(SUM(lcm.latency_ms), 0) AS llm_ms,
        COALESCE(SUM(lcm.total_tokens), 0) AS llm_tokens,
        COUNT(lcm.id) FILTER (WHERE lcm.success = false) AS failed_llm_calls,
        STRING_AGG(DISTINCT lcm.error_type, ', ')
            FILTER (WHERE lcm.error_type IS NOT NULL) AS llm_errors
    FROM react_trace rt
    LEFT JOIN llm_call_metric lcm ON lcm.trace_id = rt.id
    GROUP BY rt.request_id
),
tool_by_name AS (
    SELECT
        rt.request_id,
        tc.tool_name,
        COUNT(*) AS call_count
    FROM react_trace rt
    JOIN tool_call tc ON tc.trace_id = rt.id
    GROUP BY rt.request_id, tc.tool_name
),
tool_names AS (
    SELECT
        request_id,
        STRING_AGG(tool_name || '×' || call_count::text, ', ' ORDER BY tool_name) AS tools_used
    FROM tool_by_name
    GROUP BY request_id
),
tool_summary AS (
    SELECT
        rt.request_id,
        COUNT(tc.id) AS tool_calls,
        COALESCE(SUM(tc.latency_ms), 0) AS tool_ms,
        COALESCE(SUM(tc.upstream_call_count), 0) AS upstream_calls,
        COUNT(tc.id) FILTER (WHERE tc.status = 'ok') AS successful_tool_calls,
        COUNT(tc.id) FILTER (
            WHERE tc.id IS NOT NULL
              AND tc.status IS DISTINCT FROM 'ok'
        ) AS failed_tool_calls
    FROM react_trace rt
    LEFT JOIN tool_call tc ON tc.trace_id = rt.id
    GROUP BY rt.request_id
),
stage_summary AS (
    SELECT
        request_id,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE stage_name = 'redis_history_read'
        ), 0) AS redis_read_ms,
        COALESCE(SUM(latency_ms) FILTER (
        WHERE stage_name = 'user_context_resolution'
        ), 0) AS user_context_ms,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE stage_name = 'orchestrator'
        ), 0) AS orchestrator_ms,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE stage_name = 'grounding_verification'
        ), 0) AS grounding_ms,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE stage_name = 'redis_state_write'
        ), 0) AS redis_write_ms,
        COALESCE(SUM(latency_ms) FILTER (
            WHERE stage_name = 'db_trace_flush'
        ), 0) AS db_flush_ms,
        COUNT(*) FILTER (WHERE status = 'failed') AS failed_stages,
        STRING_AGG(DISTINCT error_type, ', ')
            FILTER (WHERE error_type IS NOT NULL) AS stage_errors
    FROM pipeline_stage_metric
    GROUP BY request_id
)
SELECT
    m.created_at,
    m.request_id,
    m.success,

    -- Полное wall-clock время запроса.
    m.total_latency_ms AS total_ms,

    -- Отдельные части pipeline.
    COALESCE(gs.scope_ms, 0) AS scope_ms,
    COALESCE(gs.input_censorship_ms, 0) AS input_censorship_ms,
    COALESCE(ss.redis_read_ms, 0) AS redis_read_ms,
    COALESCE(ss.user_context_ms, 0) AS user_context_ms,
    COALESCE(ss.orchestrator_ms, 0) AS orchestrator_ms,
    COALESCE(ls.llm_ms, 0) AS llm_ms,
    COALESCE(ts.tool_ms, 0) AS tool_ms,
    COALESCE(ss.grounding_ms, 0) AS grounding_ms,
    COALESCE(gs.output_censorship_ms, 0) AS output_censorship_ms,
    COALESCE(ss.redis_write_ms, 0) AS redis_write_ms,
    COALESCE(ss.db_flush_ms, 0) AS db_flush_ms,

    -- Количество выполненной работы.
    COALESCE(ls.llm_calls, 0) AS llm_calls,
    COALESCE(ts.tool_calls, 0) AS tool_calls,
    COALESCE(tn.tools_used, '-') AS tools_used,
    COALESCE(ts.upstream_calls, 0) AS upstream_calls,
    m.regenerations,
    COALESCE(ls.llm_tokens, 0) AS llm_tokens,

    -- Ошибки.
    COALESCE(gs.failed_gate_calls, 0) AS failed_gates,
    COALESCE(ls.failed_llm_calls, 0) AS failed_llm,
    COALESCE(ts.failed_tool_calls, 0) AS failed_tools,
    COALESCE(ss.failed_stages, 0) AS failed_stages,
    NULLIF(CONCAT_WS(
        '; ',
        NULLIF(gs.gate_errors, ''),
        NULLIF(ls.llm_errors, ''),
        NULLIF(ss.stage_errors, '')
    ), '') AS errors
FROM metrics m
LEFT JOIN gate_summary gs ON gs.request_id = m.request_id
LEFT JOIN llm_summary ls ON ls.request_id = m.request_id
LEFT JOIN tool_summary ts ON ts.request_id = m.request_id
LEFT JOIN tool_names tn ON tn.request_id = m.request_id
LEFT JOIN stage_summary ss ON ss.request_id = m.request_id
ORDER BY m.created_at DESC
LIMIT 20;
