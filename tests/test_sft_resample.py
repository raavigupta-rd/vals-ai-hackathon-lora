from poker.sft.resample import pick_reply


def response(action, reasoning="Short thought.", finish="tool_calls", tokens=300, amount=None):
    args = '{"action":"%s"%s}' % (action, f',"amount":{amount}' if amount is not None else "")
    return {"choices": [{"finish_reason": finish, "message": {"content": "\n\n", "reasoning_content": reasoning,
                                                              "tool_calls": [{"function": {"name": "submit_action", "arguments": args}}]}}],
            "usage": {"completion_tokens": tokens}}


PROMPT = [{"role": "system", "content": "rules"},
          {"role": "user", "content": "Street: preflop\nLegal actions: fold, call 2, raise to 4-200"}]


def test_pick_reply_keeps_the_first_valid_reply_with_the_teachers_action():
    samples = [response("call"), response("fold", reasoning="My own reason to fold.")]
    picked = pick_reply(PROMPT, samples, preferred="fold", validate=lambda p, r: True)
    assert picked["reasoning"] == "My own reason to fold."
    assert picked["arguments"] == {"action": "fold"}


def test_pick_reply_skips_cutoffs_overlong_and_invalid_replies():
    samples = [response("fold", finish="length"), response("fold", reasoning="x" * 20000),
               response("fold", reasoning="bad")]
    assert pick_reply(PROMPT, samples, preferred="fold",
                      validate=lambda p, r: r["choices"][0]["message"]["reasoning_content"] != "bad") is None


def test_pick_reply_accepts_bet_for_raise_and_keeps_the_amount():
    picked = pick_reply(PROMPT, [response("raise", amount=6)], preferred="raise", validate=lambda p, r: True)
    assert picked["arguments"] == {"action": "raise", "amount": 6}


from poker.sft.resample import hinted_messages, mentions_hint


def test_hint_goes_into_the_system_message_only_for_the_teacher():
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "state"}]
    hinted = hinted_messages(messages, "You need 57% equity; 96o has 38%.")
    assert hinted[0]["content"].startswith("rules") and "57%" in hinted[0]["content"]
    assert hinted[1] == messages[1] and messages[0]["content"] == "rules"  # the original is untouched


def test_replies_that_mention_the_hint_are_rejected():
    assert mentions_hint("As the coaching note says, fold.")
    assert mentions_hint("Per the hint, this is a fold.")
    assert not mentions_hint("I need 57% equity and have 38%. Fold.")
    samples = [response("fold", reasoning="The coaching note says fold.")]
    assert pick_reply(PROMPT, samples, preferred="fold", validate=lambda p, r: True) is None
