import os
import pytest
from utils import *

# #196: the adaptive MTP depth controller (draft-mtp-adaptive) restarts at its floor for every request, so the
# drafts of a request (and with them the verify widths that the temp-0 output follows) do not depend on what the
# slot served before. Needs LLAMA_TEST_MTP_MODEL_FILE (a model with an MTP head); runs on any backend.

PROMPT_A = "Write a short note about caches. " + "The cache keeps recent prompts so that repeated requests are fast. " * 20
PROMPT_B = "List the planets of the solar system in order, then describe each in one sentence: Mercury, Venus, Earth,"


def make_server(tmp_path, name: str) -> ServerProcess:
    model = os.environ.get("LLAMA_TEST_MTP_MODEL_FILE")
    if not model:
        pytest.skip("LLAMA_TEST_MTP_MODEL_FILE (a model with an MTP head) not set")
    s = ServerProcess()
    s.model_hf_repo = None
    s.model_hf_file = None
    s.model_file = model
    s.offline = True
    s.n_ctx = 4096
    s.n_slots = 1
    s.n_predict = 96
    s.temperature = 0.0
    s.spec_type = "draft-mtp-adaptive"
    s.spec_draft_n_max = 4
    s.log_path = str(tmp_path / f"server_{name}.log")
    return s


def complete(s: ServerProcess, prompt: str):
    # cache_prompt false: every request prefills the same way, so only state carried by the drafter can differ
    res = s.make_request("POST", "/completion", data={"prompt": prompt, "temperature": 0.0, "cache_prompt": False, "seed": 1})
    assert res.status_code == 200, res.body
    t = res.body["timings"]
    return res.body["content"], (t.get("draft_n"), t.get("draft_n_accepted"))


def test_spec_adaptive_depth_reset_per_request(tmp_path):
    runs = {}
    for name, order in [("boot1", ["A", "B", "A", "B", "A"]), ("boot2", ["B", "A", "A"])]:
        s = make_server(tmp_path, name)
        s.start()
        try:
            runs[name] = [(t, *complete(s, PROMPT_A if t == "A" else PROMPT_B)) for t in order]
        finally:
            s.stop()
    for name, rows in runs.items():
        print(name, [(t, d) for t, _, d in rows])
    ref = {}
    for name, rows in runs.items():
        for t, content, drafts in rows:
            assert drafts[0], "the MTP drafter did not run"
            if t not in ref:
                ref[t] = (content, drafts)
            # same content AND the same drafts: the depth trajectory restarted at the floor for this request
            assert (content, drafts) == ref[t], f"{name}: turn {t} differs from its first run"
