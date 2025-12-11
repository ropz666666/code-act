import json
import yaml
import time
import argparse
import openai
import csv


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    api = cfg.get("API Key", {})
    base_url = api.get("base_url") if isinstance(api, dict) else cfg.get("base_url")
    return {
        "openai_key": api.get("openai", ""),
        "base_url": base_url,
    }


def list_endpoints(spec_path):
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    paths = spec.get("paths", {})
    out = []
    for p, methods in paths.items():
        for m in methods.keys():
            out.append(f"{m.upper()} {p}")
    return out


def call_chat(model, messages):
    resp = openai.ChatCompletion.create(model=model, messages=messages)
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    return content, total_tokens


def predict_plan(model, requirement, endpoints):
    prompt = (
        "You are an API planner. Given a user requirement and a list of available endpoints, "
        "produce the minimal ordered sequence of endpoints to fulfill the requirement. "
        "Only use endpoints from the list. Respond strictly as JSON with a 'plan' array of strings."
    )
    ep_text = "\n".join(endpoints)
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Requirement:\n{requirement}\n\nEndpoints:\n{ep_text}\n\nReturn JSON with key 'plan'.",
        },
    ]
    content, toks = call_chat(model, messages)
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        obj = json.loads(content[start:end])
        plan = obj.get("plan", [])
    except Exception:
        plan = []
    return plan, toks


def generate_code(model, requirement, plan):
    plan_text = "\n".join(plan)
    sys_msg = (
        "Write Python code using requests to execute the given API plan against TMDB API. "
        "Return only code."
    )
    user_msg = f"Requirement:\n{requirement}\n\nPlan:\n{plan_text}"
    content, toks = call_chat(model, [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}])
    return content, toks


def judge_success(model, requirement, plan, solution):
    user_msg = (
        "Judge if the predicted plan satisfies the requirement. "
        "Return JSON with key 'success' as 1 or 0.\n\n"
        f"Requirement:\n{requirement}\n\nPredicted plan:\n{plan}\n\nReference solution:\n{solution}"
    )
    content, toks = call_chat(model, [{"role": "user", "content": user_msg}])
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        obj = json.loads(content[start:end])
        succ = int(obj.get("success", 0))
        succ = 1 if succ == 1 else 0
    except Exception:
        succ = 0
    return succ, toks


def run(dataset_path, spec_path, config_path, out_path, limit, model_name):
    cfg = load_config(config_path)
    openai.api_base = cfg["base_url"]
    openai.api_key = cfg["openai_key"]
    endpoints = list_endpoints(spec_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for i, item in enumerate(data):
        if limit and i >= limit:
            break
        req = item.get("query", "")
        sol = item.get("solution", [])
        t0 = time.time()
        llm_calls = 0
        token_cost = 0
        error = ""
        try:
            plan, t1 = predict_plan(model_name, req, endpoints)
            llm_calls += 1
            token_cost += t1
            code, t2 = generate_code(model_name, req, plan)
            llm_calls += 1
            token_cost += t2
            success, t3 = judge_success(model_name, req, plan, sol)
            llm_calls += 1
            token_cost += t3
            result = 1 if plan == sol else 0
            pass_rate = 1 if result == 1 else 0
            cost_in_time = round(time.time() - t0, 3)
            rows.append(
                {
                    "requirement": req,
                    "pass_rate": pass_rate,
                    "success_rate": success,
                    "llm_calls": llm_calls,
                    "token_cost": token_cost,
                    "runnable_code": code,
                    "cost_in_time": cost_in_time,
                    "result": result,
                    "error": error,
                }
            )
        except Exception as e:
            cost_in_time = round(time.time() - t0, 3)
            rows.append(
                {
                    "requirement": req,
                    "pass_rate": 0,
                    "success_rate": 0,
                    "llm_calls": llm_calls,
                    "token_cost": token_cost,
                    "runnable_code": "",
                    "cost_in_time": cost_in_time,
                    "result": 0,
                    "error": str(e),
                }
            )
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "requirement",
                "pass_rate",
                "success_rate",
                "llm_calls",
                "token_cost",
                "runnable_code",
                "cost_in_time",
                "result",
                "error",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./experiment/datasets/tmdb.json")
    p.add_argument("--spec", default="./experiment/specs/tmdb_oas.json")
    p.add_argument("--config", default="./config.yaml")
    p.add_argument("--out", default="./experiment/result/tmdb.csv")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--model", default="gpt-3.5-turbo-0613")
    args = p.parse_args()
    run(args.dataset, args.spec, args.config, args.out, args.limit, args.model)


if __name__ == "__main__":
    main()
