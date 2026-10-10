#!/usr/bin/env python3
"""Official 108-cell response-generation runner for the prompt-invariance experiment."""

from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MASTER = Path("prompts_master_final.jsonl")
DEFAULT_OUTPUT = Path("responses_official.jsonl")
DEFAULT_SUMMARY = Path("generation_summary.csv")
DEFAULT_MANIFEST = Path("generation_manifest.json")
MAX_TOKENS = 4096
TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = [2, 5]

MODELS = [
    {"key":"deepseek_v3","name":"DeepSeek V3","id":"deepseek/deepseek-chat","provider":"novita","provider_name":"NovitaAI","temperature":0.0,"reasoning":None},
    {"key":"deepseek_r1","name":"DeepSeek R1","id":"deepseek/deepseek-r1","provider":"novita","provider_name":"NovitaAI","temperature":0.0,"reasoning":None},
    {"key":"gpt4o","name":"GPT-4o","id":"openai/gpt-4o","provider":"openai","provider_name":"OpenAI","temperature":0.0,"reasoning":None},
    {"key":"o3_mini","name":"o3-mini","id":"openai/o3-mini","provider":"openai","provider_name":"OpenAI","temperature":None,"reasoning":{"effort":"medium"}},
]


def now(): return datetime.now(timezone.utc).isoformat()

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def load_jsonl(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON at {path}:{n}: {e}") from e
    return rows

def append_jsonl(path,obj):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False)+"\n"); f.flush(); os.fsync(f.fileno())

def validate_master(rows):
    if len(rows)!=27: raise ValueError(f"Expected 27 master rows; found {len(rows)}")
    expected={"C0_base","C1_character_swap","C2_lexical_substitution"}
    if sorted({r["family_id"] for r in rows})!=list(range(9)): raise ValueError("Expected family_id 0..8")
    required={"family_id","source_instance_idx","base_id","variant_id","condition_code","condition_label","capability","task","system_prompt","input","reference_answer","score_rubric"}
    for i,r in enumerate(rows):
        missing=required-set(r)
        if missing: raise ValueError(f"Row {i} missing fields: {sorted(missing)}")
    for fid in range(9):
        fam=[r for r in rows if r["family_id"]==fid]
        if len(fam)!=3 or {r["condition_code"] for r in fam}!=expected: raise ValueError(f"Bad triplet for family {fid}")
        for field in ("base_id","input","reference_answer","score_rubric","task","capability"):
            vals=[r[field] for r in fam]
            if not (vals[0]==vals[1]==vals[2]): raise ValueError(f"family {fid}: {field} differs across conditions")

def cell_id(row,model): return f"{row['variant_id']}::{model['key']}"

def existing_statuses(path):
    out={}
    if Path(path).exists():
        for r in load_jsonl(path): out[r.get("generation_cell_id")]=r.get("generation_status")
    return out

def classify(code):
    if code==408: return "timeout_http_408",True
    if code==429: return "rate_limited",True
    if 500<=code<=599: return "server_error",True
    if 400<=code<=499: return "client_error",False
    return "http_error",False

def sanitize(msg,key):
    if msg is None: return None
    msg=str(msg).replace(key,"[REDACTED_API_KEY]") if key else str(msg)
    if "Bearer " in msg: msg=msg.split("Bearer ",1)[0]+"Bearer [REDACTED]"
    return msg[:4000]

def payload_for(row,model):
    p={
        "model":model["id"],
        "messages":[{"role":"system","content":row["system_prompt"]},{"role":"user","content":row["input"]}],
        "max_tokens":MAX_TOKENS,
        "provider":{"only":[model["provider"]],"allow_fallbacks":False,"require_parameters":True},
    }
    if model["temperature"] is not None: p["temperature"]=model["temperature"]
    if model["reasoning"] is not None: p["reasoning"]=model["reasoning"]
    return p

def one_attempt(session,headers,payload,n,key):
    t0=time.monotonic(); log={"attempt_number":n,"started_at_utc":now(),"ended_at_utc":None,"elapsed_seconds":None,"outcome":None,"http_status":None,"retryable":None,"error_type":None,"error_message":None}
    try:
        resp=session.post(URL,headers=headers,json=payload,timeout=TIMEOUT_SECONDS)
        log["ended_at_utc"]=now(); log["elapsed_seconds"]=round(time.monotonic()-t0,3); log["http_status"]=resp.status_code
        if resp.status_code>=400:
            outcome,retry=classify(resp.status_code); log["outcome"]=outcome; log["retryable"]=retry
            try: raw=json.dumps(resp.json(),ensure_ascii=False)
            except Exception: raw=resp.text
            log["error_message"]=sanitize(raw,key); return log,None
        try: data=resp.json()
        except ValueError as e:
            log.update(outcome="invalid_json_response",retryable=True,error_type=type(e).__name__,error_message=sanitize(str(e),key)); return log,None
        if not data.get("choices"):
            log.update(outcome="missing_choices",retryable=True,error_message=sanitize(json.dumps(data,ensure_ascii=False),key)); return log,None
        log.update(outcome="success",retryable=False); return log,data
    except requests.Timeout as e:
        log.update(ended_at_utc=now(),elapsed_seconds=round(time.monotonic()-t0,3),outcome="timeout",retryable=True,error_type=type(e).__name__,error_message=sanitize(str(e),key)); return log,None
    except requests.ConnectionError as e:
        log.update(ended_at_utc=now(),elapsed_seconds=round(time.monotonic()-t0,3),outcome="connection_error",retryable=True,error_type=type(e).__name__,error_message=sanitize(str(e),key)); return log,None
    except requests.RequestException as e:
        log.update(ended_at_utc=now(),elapsed_seconds=round(time.monotonic()-t0,3),outcome="request_error",retryable=False,error_type=type(e).__name__,error_message=sanitize(str(e),key)); return log,None

def usage_from(data):
    u=data.get("usage") or {}; d=u.get("completion_tokens_details") or {}
    reasoning=d.get("reasoning_tokens")
    if reasoning is None: reasoning=u.get("reasoning_tokens") or d.get("reasoning")
    return {"prompt_tokens":u.get("prompt_tokens"),"completion_tokens":u.get("completion_tokens"),"reasoning_tokens":reasoning,"total_tokens":u.get("total_tokens"),"reported_cost":u.get("cost")}

def generate_cell(session,headers,row,model,key):
    attempts=[]; data=None; payload=payload_for(row,model)
    for n in range(1,MAX_ATTEMPTS+1):
        log,data=one_attempt(session,headers,payload,n,key); attempts.append(log)
        if data is not None or not log.get("retryable",False): break
        if n<MAX_ATTEMPTS: time.sleep(BACKOFF_SECONDS[min(n-1,len(BACKOFF_SECONDS)-1)])
    result={
        "generation_cell_id":cell_id(row,model),"family_id":row["family_id"],"source_instance_idx":row["source_instance_idx"],"base_id":row["base_id"],"variant_id":row["variant_id"],"condition_code":row["condition_code"],"condition_label":row["condition_label"],"capability":row["capability"],"task":row["task"],
        "system_prompt":row["system_prompt"],"input":row["input"],"reference_answer":row["reference_answer"],"score_rubric":row["score_rubric"],
        "response_model_key":model["key"],"response_model_display":model["name"],"requested_model_id":model["id"],"requested_provider_slug":model["provider"],"requested_provider_display":model["provider_name"],
        "temperature_requested":model["temperature"],"temperature_applied":model["temperature"],"temperature_note":None if model["temperature"] is not None else "Temperature intentionally omitted for o3-mini.","reasoning_config":model["reasoning"],
        "max_completion_tokens":MAX_TOKENS,"request_timeout_seconds":TIMEOUT_SECONDS,"max_attempts":MAX_ATTEMPTS,
        "generation_status":None,"successful_attempt_number":None,"attempts":attempts,"response_text":None,"finish_reason":None,"truncated":None,"empty_response":None,
        "prompt_tokens":None,"completion_tokens":None,"reasoning_tokens":None,"total_tokens":None,"reported_cost":None,
        "returned_model_id":None,"returned_provider":None,"openrouter_response_id":None,"created":None,"completed_at_utc":now(),
    }
    if data is None:
        result["generation_status"]="unresolved_after_max_attempts"; return result
    choice=data["choices"][0]; message=choice.get("message") or {}; finish=choice.get("finish_reason"); text=message.get("content")
    result.update({"generation_status":"success","successful_attempt_number":next(a["attempt_number"] for a in attempts if a["outcome"]=="success"),"response_text":text,"finish_reason":finish,"truncated":finish in {"length","max_tokens"},"empty_response":text is None or (isinstance(text,str) and text.strip()==""),**usage_from(data),"returned_model_id":data.get("model"),"returned_provider":data.get("provider"),"openrouter_response_id":data.get("id"),"created":data.get("created")})
    return result

SUMMARY_FIELDS=["generation_cell_id","family_id","source_instance_idx","base_id","variant_id","condition_code","condition_label","capability","task","response_model_key","response_model_display","requested_model_id","requested_provider_display","temperature_applied","reasoning_config","max_completion_tokens","generation_status","successful_attempt_number","finish_reason","truncated","empty_response","prompt_tokens","completion_tokens","reasoning_tokens","total_tokens","reported_cost","returned_model_id","returned_provider","openrouter_response_id","completed_at_utc"]

def rebuild_summary(jsonl_path,csv_path):
    rows=load_jsonl(jsonl_path) if Path(jsonl_path).exists() else []
    with open(csv_path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=SUMMARY_FIELDS); w.writeheader()
        for r in rows:
            out={k:r.get(k) for k in SUMMARY_FIELDS}
            if out.get("reasoning_config") is not None: out["reasoning_config"]=json.dumps(out["reasoning_config"],sort_keys=True)
            w.writerow(out)

def write_manifest(path,master,script):
    m={
        "experiment_stage":"official_response_generation","created_at_utc":now(),"master_file":str(master),"master_file_sha256":sha256_file(master),"master_rows":27,"families":9,"conditions":["C0_base","C1_character_swap","C2_lexical_substitution"],"response_models":4,"expected_generation_cells":108,
        "openrouter_endpoint":URL,"provider_fallbacks_allowed":False,"max_completion_tokens":MAX_TOKENS,"request_timeout_seconds":TIMEOUT_SECONDS,"max_attempts":MAX_ATTEMPTS,"models":MODELS,
        "pilot_decisions":{"deepseek_r1_temperature":0.0,"r1_selection_basis":"Retained after dual-temperature pilot (0.0 vs 0.6) found no targeted repetition/looping/incoherence pathology at temperature 0.","token_cap_basis":"Universal completion cap fixed prospectively at 4096 after retained pilot configurations had maximum observed completion usage of 2097 tokens.","pilot_data_excluded_from_primary_analysis":True},
        "retry_policy":{"successful_cells_regenerated_automatically":False,"retryable":["timeout","HTTP 408","HTTP 429","HTTP 5xx","connection_error","invalid_json_response","missing_choices"]},
        "environment":{"python_version":sys.version,"platform":platform.platform(),"requests_version":getattr(requests,"__version__",None)},"script_file":str(script),"script_file_sha256":sha256_file(script) if Path(script).exists() else None,
    }
    Path(path).write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding="utf-8")

def audit_output(path):
    rows=load_jsonl(path) if Path(path).exists() else []; success=[r for r in rows if r.get("generation_status")=="success"]
    return {"rows":len(rows),"success":len(success),"unresolved":len(rows)-len(success),"truncated":sum(bool(r.get("truncated")) for r in success),"empty":sum(bool(r.get("empty_response")) for r in success),"retry_success":sum((r.get("successful_attempt_number") or 0)>1 for r in success),"cost":sum(float(r.get("reported_cost") or 0) for r in success)}

def main():
    ap=argparse.ArgumentParser(description="Generate the 108 official response cells.")
    ap.add_argument("--master",type=Path,default=DEFAULT_MASTER); ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); ap.add_argument("--summary",type=Path,default=DEFAULT_SUMMARY); ap.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST); ap.add_argument("--retry-unresolved",action="store_true")
    args=ap.parse_args()
    key=os.environ.get("OPENROUTER_API_KEY","").strip()
    if not key: sys.exit("OPENROUTER_API_KEY is not set.")
    if not args.master.exists(): sys.exit(f"Master file not found: {args.master}")
    rows=load_jsonl(args.master); validate_master(rows); write_manifest(args.manifest,args.master,Path(__file__).resolve()); existing=existing_statuses(args.output)
    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json","X-Title":"Prompt Invariance Official Generation"}
    session=requests.Session(); total=len(rows)*len(MODELS); i=0
    print("Official generation configuration validated."); print("27 prompt variants x 4 models = 108 official cells."); print(f"max_tokens={MAX_TOKENS}; timeout={TIMEOUT_SECONDS}s; max_attempts={MAX_ATTEMPTS}\n")
    for row in rows:
        for model in MODELS:
            i+=1; cid=cell_id(row,model); prior=existing.get(cid)
            if prior=="success": print(f"[{i:03}/{total}] SKIP successful {cid}"); continue
            if prior is not None and not args.retry_unresolved: print(f"[{i:03}/{total}] SKIP prior={prior} {cid}"); continue
            print(f"[{i:03}/{total}] RUN family={row['family_id']} condition={row['condition_code']} model={model['name']}")
            result=generate_cell(session,headers,row,model,key); append_jsonl(args.output,result); existing[cid]=result["generation_status"]; rebuild_summary(args.output,args.summary)
            if result["generation_status"]=="success": print(f"    success | completion_tokens={result['completion_tokens']} | reasoning_tokens={result['reasoning_tokens']} | finish={result['finish_reason']} | attempt={result['successful_attempt_number']}")
            else: print(f"    unresolved | attempts={len(result['attempts'])}")
    rebuild_summary(args.output,args.summary); a=audit_output(args.output)
    print(f"\nRows={a['rows']} success={a['success']} unresolved={a['unresolved']} truncated={a['truncated']} empty={a['empty']} retry_success={a['retry_success']} cost=${a['cost']:.6f}")
    print(f"Official responses: {args.output}\nGeneration summary: {args.summary}\nGeneration manifest: {args.manifest}")
    if a["rows"]==108 and a["success"]==108: print("INTEGRITY CHECK: 108/108 successful. Freeze responses_official.jsonl before autograding.")
    else: print("INTEGRITY CHECK INCOMPLETE: do not begin autograding until 108/108 cells succeed.")

if __name__=="__main__": main()
