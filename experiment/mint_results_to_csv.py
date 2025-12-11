import json
import csv
import argparse


def parse_llm_calls(state):
    cnt = sum(state.get("agent_action_count", {}).values())
    hist = state.get("history", [])
    fb = 0
    for h in hist:
        if h.get("role") == "user" and "Expert feedback:" in h.get("content", ""):
            fb += 1
    return cnt + fb


def parse_token_cost(token_counter):
    return sum(token_counter.values())


def parse_runnable_code(state):
    for h in reversed(state.get("history", [])):
        if h.get("role") == "assistant":
            c = h.get("content", "")
            if "<execute>" in c:
                s = c.split("<execute>")[-1]
                s = s.split("</execute>")[0]
                return s.strip()
            if "<solution>" in c:
                s = c.split("<solution>")[-1]
                s = s.split("</solution>")[0]
                return s.strip()
    return ""


def parse_success_rate(state):
    fb = state.get("latest_output", {}).get("feedback", "")
    if "GOOD" in fb:
        return 1
    if "BAD" in fb:
        return 0
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            state = obj["state"]
            task = obj["task"]
            requirement = task.get("prompt", "")
            pass_rate = 1 if state.get("success", False) else 0
            success_rate = parse_success_rate(state)
            llm_calls = parse_llm_calls(state)
            token_cost = parse_token_cost(state.get("token_counter", {}))
            runnable_code = parse_runnable_code(state)
            duration = state.get("latest_output", {}).get("duration_sec", None)
            result = pass_rate
            error = state.get("error", "") or ""
            rows.append({
                "requirement": requirement,
                "pass_rate": pass_rate,
                "success_rate": success_rate,
                "llm_calls": llm_calls,
                "token_cost": token_cost,
                "runnable_code": runnable_code,
                "cost_in_time": duration,
                "result": result,
                "error": error,
            })

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "requirement",
            "pass_rate",
            "success_rate",
            "llm_calls",
            "token_cost",
            "runnable_code",
            "cost_in_time",
            "result",
            "error",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()

