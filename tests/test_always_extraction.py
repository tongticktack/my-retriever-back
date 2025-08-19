from app.services import extraction_controller as extc
from app.services import lost_item_extractor as li_ext


def test_always_runs_extraction(monkeypatch):
    state = {"items": [{"extracted": {}, "missing": li_ext.compute_missing({}), "stage": "collecting"}], "active_index": 0}
    called = {"llm": 0}
    def fake_llm(user_text, current):
        called["llm"] += 1
        return {}
    monkeypatch.setattr(extc, '_strict_llm_extract', fake_llm)
    # two normal info messages
    extc.process_message("강남에서 빨간 가방 잃어버렸어요", state, False)
    extc.process_message("어제 오후쯤입니다", state, False)
    assert called["llm"] == 2
    # confirm token should not trigger extraction
    state['items'][0]['stage'] = 'ready'
    extc.process_message("예", state, False)
    assert called["llm"] == 2
