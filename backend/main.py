# main.py
import os
import io
import json
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Dict, Any, List
import logging
import traceback

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import tiktoken

try:
    from openai import OpenAI
    OPENAI_INSTALLED = True
except Exception:
    OpenAI = None
    OPENAI_INSTALLED = False

try:
    import ijson
    HAVE_IJSON = True
except Exception:
    HAVE_IJSON = False

app = FastAPI(title="Log Analysis - single OpenAI call")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("relvy")

# models and configs
MODEL = "gpt-4o-mini"
COST = {"input": 0.3, "output": 2.4}  

MAX_LOGS_TO_SCAN = 500_000
MAX_FILTERED_TO_STORE = 2000
MAX_LLM_LOGS = 200  

# Single-call token budget
TOKENS_PER_REQUEST = 32000    
EXPECTED_OUTPUT_TOKENS = 800   
SAFETY_MARGIN = 200
BODY_PREVIEW_LEN = 300
MAX_FULL_BODY_STORE = 1000

_full_body_store: Dict[str, str] = {}
executor = ThreadPoolExecutor(max_workers=4)

OPENAI_API_KEY = '<input_your_key_here>'  # os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_INSTALLED and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------- Helpers ----------------------
def normalize_log(entry: dict) -> dict:
    resource = entry.get("resource_attributes", {}) if isinstance(entry, dict) else {}
    fields = entry.get("fields", {}) if isinstance(entry, dict) else {}
    return {
        "timestamp_raw": entry.get("timestamp") or entry.get("time") or entry.get("ts"),
        "container": (resource.get("k8s.container.name") or resource.get("k8s.deployment.name")
                      or entry.get("containerName") or entry.get("container")),
        "service": resource.get("service.name") or entry.get("service") or entry.get("containerName"),
        "severity": fields.get("severity_text") or entry.get("level") or entry.get("severity"),
        "body": entry.get("body") or entry.get("log") or entry.get("message") or "",
        "_raw": entry
    }

def keywords_from_prompt(prompt: str) -> List[str]:
    return [k.strip().lower() for k in prompt.split() if k.strip()]

def entry_matches(entry: dict, keywords: List[str]) -> bool:
    if not keywords:
        return True
    hay = " ".join([str(entry.get("container","") or ""), str(entry.get("service","") or ""), str(entry.get("body","") or "")]).lower()
    return any(k in hay for k in keywords)

def preview_body(body: str, length: int = BODY_PREVIEW_LEN) -> str:
    if not isinstance(body, str):
        body = str(body)
    return body if len(body) <= length else body[:length] + "..."

def make_id(log: dict) -> str:
    body = (log.get("body") or "") + str(log.get("timestamp_raw") or "")
    return hashlib.sha1(body.encode("utf-8", errors="ignore")).hexdigest()[:12]

def ns_to_iso(ns) -> str | None:
    try:
        ns_int = int(ns)
        s, ns_remainder = divmod(ns_int, 1_000_000_000)
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(s, tz=timezone.utc).replace(microsecond=ns_remainder // 1000)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return None

# Iterators for NDJSON / JSON array streaming
def iter_ndjson_lines(file_obj: io.BufferedReader) -> Iterator[dict]:
    for raw_line in file_obj:
        if not raw_line:
            break
        try:
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8", errors="replace")
            else:
                line = raw_line
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
        except Exception:
            continue

def iter_json_array_stream(file_obj: io.BufferedReader) -> Iterator[dict]:
    for item in ijson.items(file_obj, "item"):
        yield item

def estimate_tokens_for_text(text: str, model: str = MODEL) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Compact logs: group identical bodies and keep counts + preview to reduce token size
def compact_logs_for_single_call(logs: List[dict], max_items: int = MAX_LLM_LOGS) -> List[dict]:
    # group by body content (trimmed), keep count and endpoints
    groups: Dict[str, Dict[str, Any]] = {}
    for l in logs:
        body = (l.get("body") or "").strip()
        if not body:
            body = "<empty>"
        key = hashlib.sha1(body.encode("utf-8", errors="ignore")).hexdigest()
        if key in groups:
            groups[key]["count"] += 1
            # update last_seen timestamp if available
            ts = l.get("timestamp_raw")
            if ts:
                groups[key]["last_seen"] = ns_to_iso(ts) or groups[key]["last_seen"]
        else:
            groups[key] = {
                "id": make_id(l),
                "container": l.get("container"),
                "service": l.get("service"),
                "severity": l.get("severity"),
                "body_preview": preview_body(body, BODY_PREVIEW_LEN),
                "body_truncated": len(body) > BODY_PREVIEW_LEN,
                "count": 1,
                "first_seen": ns_to_iso(l.get("timestamp_raw")),
                "last_seen": ns_to_iso(l.get("timestamp_raw"))
            }
    # convert to list and sort by count desc
    grouped = sorted(groups.values(), key=lambda x: x["count"], reverse=True)
    return grouped[:max_items]

# Single blocking call wrapper for v1 client (chat.completions)
def call_openai_single_sync(client_obj, payload: dict) -> dict:
    if client_obj is None:
        raise RuntimeError("OpenAI client not initialized (OPENAI_API_KEY missing or client failed).")
    # Use chat completions v1 method
    return client_obj.chat.completions.create(**payload)

# ---------------------- Endpoint ----------------------
@app.post("/analyze/")
async def analyze(file: UploadFile, prompt: str = Form(...)):
    DEV = os.getenv("DEV", "false").lower() in ("1", "true", "yes")
    try:
        # basic checks
        if not OPENAI_INSTALLED:
            raise HTTPException(status_code=500, detail="openai (v1+) package not installed. pip install openai>=1.0.0")
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment.")

        # size guard
        try:
            file.file.seek(0, io.SEEK_END)
            size = file.file.tell()
            file.file.seek(0)
        except Exception:
            size = None

        if size is not None and size > 200 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (>200MB). Pre-filter or split the upload.")

        keywords = keywords_from_prompt(prompt)
        filtered: List[dict] = []
        total_scanned = 0
        total_matched = 0

        # detect format
        first_chunk = file.file.read(1024)
        if isinstance(first_chunk, bytes):
            first_chunk_str = first_chunk.decode("utf-8", errors="ignore")
        else:
            first_chunk_str = str(first_chunk or "")
        file.file.seek(0)

        try:
            if first_chunk_str.lstrip().startswith("[") and HAVE_IJSON:
                iterator = iter_json_array_stream(file.file)
            else:
                iterator = iter_ndjson_lines(file.file)
        except Exception:
            file.file.seek(0)
            iterator = iter_ndjson_lines(file.file)

        for entry in iterator:
            total_scanned += 1
            if total_scanned > MAX_LOGS_TO_SCAN:
                break
            nl = normalize_log(entry)
            if entry_matches(nl, keywords):
                total_matched += 1
                if len(filtered) < MAX_FILTERED_TO_STORE:
                    filtered.append(nl)

        # pick initial set to consider (larger to allow compaction)
        candidate_logs = filtered[:MAX_LLM_LOGS]
        # Create a "detailed" representation for the single call
        detailed_logs_text = json.dumps(candidate_logs, indent=2, ensure_ascii=False)

        base_prompt = (
            f"Incident: {prompt}\n"
            "Task: From the logs provided, identify logs most relevant to the incident.\n"
            "For each highlighted log return a JSON object with keys: id, reason (1-2 sentences), score (0-1), and timestamp and user id.\n"
            "Return ONLY a JSON array of such objects. Keep output strictly machine-parseable and concise.\n\n"
            "Logs (detailed):\n"
        )

        # Estimate tokens for the detailed payload
        full_input_text = base_prompt + detailed_logs_text
        input_tokens_est = estimate_tokens_for_text(full_input_text, MODEL)
        max_allowed = TOKENS_PER_REQUEST - EXPECTED_OUTPUT_TOKENS - SAFETY_MARGIN

        # If too big, compact the logs and re-estimate
        used_compaction = False
        compacted_payload_text = None
        if input_tokens_est > max_allowed:
            used_compaction = True
            compacted_logs = compact_logs_for_single_call(candidate_logs, max_items=MAX_LLM_LOGS)
            compacted_logs_text = json.dumps(compacted_logs, indent=2, ensure_ascii=False)
            compact_prompt = (
                f"Incident: {prompt}\n"
                "Task: From the log summaries provided (grouped with counts and previews), identify logs most relevant to the incident.\n"
                "For each highlighted log return a JSON object with keys: id, reason (1-2 sentences), score (0-1).\n"
                "Return ONLY a JSON array of such objects. Keep output strictly machine-parseable and concise.\n\n"
                "Log Summaries:\n"
            )
            compacted_payload_text = compact_prompt + compacted_logs_text
            input_tokens_est = estimate_tokens_for_text(compacted_payload_text, MODEL)

        # If still too big, refuse to call and ask user to reduce logs
        if input_tokens_est > max_allowed:
            return JSONResponse(status_code=400, content={
                "error": "input_too_large",
                "message": "Even compacted logs exceed token budget for a single LLM call. Please pre-filter logs or reduce file size."
            })

        # Build final content (compacted if used)
        final_content = compacted_payload_text if used_compaction else full_input_text

        # Compute input cost estimate
        input_cost_est = (input_tokens_est / 1_000_000) * COST["input"]

        # Prepare payload for single API call
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a concise log analyst."},
                {"role": "user", "content": final_content}
            ],
            "max_tokens": EXPECTED_OUTPUT_TOKENS,
            "temperature": 0.0
        }

        # Log debug
        logger.info(f"Single OpenAI call prepared: tokens_est={input_tokens_est}, max_allowed={max_allowed}, used_compaction={used_compaction}")

        # Make exactly one OpenAI API call (no retries)
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(executor, call_openai_single_sync, client, payload)
        except Exception as e:
            # Attach API error details; if DEV, include traceback
            tb = traceback.format_exc()
            logger.error("OpenAI single call failed: %s\n%s", e, tb)
            detail = str(e)
            content = {"error": "openai_call_failed", "message": detail}
            if DEV:
                content["traceback"] = tb
            return JSONResponse(status_code=500, content=content)

        # Parse model output (try strict JSON parse)
        out_text = ""
        try:
            # v1 client returns choices; handle dict-like or object-like shapes
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                if choices and isinstance(choices, list):
                    out_text = choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
                else:
                    out_text = str(resp)
                usage = resp.get("usage", {}) or {}
                out_toks = usage.get("completion_tokens") or usage.get("total_tokens") or 0
            else:
                # object-like response
                choices = getattr(resp, "choices", None)
                if choices:
                    first = choices[0]
                    # try attribute access
                    out_text = getattr(getattr(first, "message", None), "content", None) or getattr(first, "text", None) or str(first)
                else:
                    out_text = str(resp)
                usage = getattr(resp, "usage", {}) or {}
                out_toks = usage.get("completion_tokens") or usage.get("total_tokens") or 0
        except Exception:
            out_text = str(resp)
            out_toks = 0

        # Attempt to parse JSON array from model output
        highlighted = []
        try:
            parsed = json.loads(out_text.strip())
            if isinstance(parsed, list):
                highlighted = parsed
            else:
                # unexpected shape: wrap it
                highlighted = [{"raw_text": parsed}]
        except Exception:
            # try to extract first JSON array substring
            # Extract text from first choice
            try:
                output_text = resp.choices[0].message.content.strip()
            except Exception:
                output_text = str(resp)

            # If wrapped in ```json fences, strip them
            if output_text.startswith("```"):
                output_text = output_text.strip("`").lstrip("json").strip()

            # Try to parse as JSON
            highlighted = []
            try:
                parsed = json.loads(output_text)
                if isinstance(parsed, list):
                    highlighted = parsed
            except Exception:
                # fallback: just keep raw text
                parsed_items = [{"raw_text": output_text}]

        # Estimate output tokens if out_toks missing
        try:
            total_output_tokens = out_toks if out_toks else estimate_tokens_for_text(out_text, MODEL)
        except Exception:
            total_output_tokens = 0

        total_input_tokens = input_tokens_est

        input_cost = (total_input_tokens / 1_000_000) * COST["input"]
        output_cost = (total_output_tokens / 1_000_000) * COST["output"]

        # Prepare filtered sample for UI with previews and ids
        filtered_sample = []
        # use candidate_logs (pre-compaction content) for sample (but limited)
        for nl in candidate_logs:
            entry_id = make_id(nl)
            if len(_full_body_store) < MAX_FULL_BODY_STORE:
                _full_body_store[entry_id] = nl.get("body","")
            filtered_sample.append({
                "id": entry_id,
                "timestamp_iso": ns_to_iso(nl.get("timestamp_raw")),
                "container": nl.get("container"),
                "service": nl.get("service"),
                "severity": nl.get("severity"),
                "body_preview": preview_body(nl.get("body","")),
                "body_truncated": len(str(nl.get("body",""))) > BODY_PREVIEW_LEN
            })

        response_payload = {
            "schema_version": "1.3",
            "summary": {
                "total_scanned": total_scanned,
                "total_matched": total_matched,
                "kept_filtered": len(filtered),
                "llm_sent_logs": len(candidate_logs),
                "used_compaction": used_compaction
            },
            "filtered_logs_sample": filtered_sample,
            "highlighted_logs": highlighted,
            "cost": {
                "input": round(input_cost, 6),
                "output": round(output_cost, 6),
                "total": round(input_cost + output_cost, 6),
                "tokens": {"input_tokens": total_input_tokens, "output_tokens": total_output_tokens},
                "note": "pricing per 1M tokens (estimates)"
            }
        }

        return JSONResponse(response_payload)

    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unhandled exception in /analyze/: %s\n%s", exc, tb)
        error_payload = {"error": "internal_server_error", "message": str(exc)}
        if DEV:
            error_payload["traceback"] = tb
        return JSONResponse(status_code=500, content=error_payload)

# Lazy fetch full body
@app.get("/log/{log_id}")
async def get_full_log(log_id: str):
    body = _full_body_store.get(log_id)
    if body is None:
        raise HTTPException(status_code=404, detail="Full body not available (may have been dropped to save memory)")
    return {"id": log_id, "body": body}

@app.get("/health")
async def health():
    return {"status": "ok"}
