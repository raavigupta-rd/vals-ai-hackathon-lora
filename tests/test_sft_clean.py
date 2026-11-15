import json

from poker.sft.clean import add_pad_read_line, clean

SYSTEM = {"role": "system", "content": "rules"}


def pad(hand=10, vb=None, lines=()):
    vb = hand if vb is None else vb
    head = f"H{hand}/30 | me seat4 354 (177 BB) | rank 1/5 | gap up +0, down -108 | {30 - hand} left"
    opps = [f"seat{s}: 200 | VPIP 3/{vb} | open 1/2 | 3bet 0/5 | jams 0 | folds-to-raise 5/5 | shown: -" for s in (0, 1, 2, 5)]
    return "\n".join([head, *opps, "blinds: fold-to-steal 5/5", "out: seat3@H1", *lines])


FULL_PAD = pad(lines=[
    "> survive: leading late, avoid big pots with marginal hands",
    "> blinds fold to steals (5/5): open BTN/CO wide",
    "> seat0 folds to raises (5/5): raise first-in and 3-bet light",
    "> seat1 is tight (VPIP 1/10): respect its raises",
    "> seat5 calls a lot: value bet, do not bluff",
])


def ring(hero, button, folded=(), out=(3,), bets=None):
    bets = bets or {}
    lines = []
    for s in range(6):
        tag = " [button]" if s == button else ""
        you = " (you)" if s == hero else ""
        status = ", out" if s in out else (", folded" if s in folded else "")
        stack = 0 if s in out else 200
        lines.append(f"  seat {s}: seat{s}{tag}{you} stack {stack}, bet {bets.get(s, 0)}{status}")
    return lines


def messages(*, hand=10, pad_text=FULL_PAD, seat=2, position=2, button=5, hole="Qh Tc", board="-", street="preflop",
             pot=3, legal="fold, call 2, raise to 4-200", events=(), folded=(), bets=None):
    """Live ring 0,1,2,4,5 (seat 3 out), button seat 5: SB 0, BB 1, UTG 2 (position 2), CO 4 (3), BTN 5 (4)."""
    first = f"## Your scratchpad (private: only you can see it)\n<<<SCRATCHPAD\n{pad_text}\nSCRATCHPAD>>>\n752/5000 characters."
    table = ring(seat, button, folded=folded, bets=bets or {0: 1, 1: 2})
    acts = "\n".join(f"  {e}" for e in events) or "  (no actions yet)"
    body = (f"## Hand {hand} of 30\nYour seat: {seat} (position {position}; 0 = small blind)\nStreet: {street}\n"
            f"Your hole cards: {hole}\nBoard: {board}\nPot (including current bets): {pot}\nTable:\n" + "\n".join(table)
            + f"\nActions this hand:\n{acts}\n\nLegal actions: {legal}\nCall submit_action with your decision.")
    return [SYSTEM, {"role": "user", "content": first}, {"role": "user", "content": body}]


def record(msgs, action="fold", amount=None, reasoning="Weak hand, fold.", source="self_clean", content=""):
    args = {"action": action} | ({"amount": amount} if amount is not None else {})
    return {"messages": msgs, "weight": 1.0, "source": source,
            "target": {"reasoning": reasoning, "content": content,
                       "tool_call": {"name": "submit_action", "arguments": args}}}


def pad_record(hand, contents, source="pad_teacher"):
    over = f"## Hand {hand} is over\nSince your last action:\n  (no other actions)\n\nNow call update_scratchpad exactly once."
    return {"messages": [SYSTEM, {"role": "user", "content": "x"}, {"role": "user", "content": over}], "weight": 1.0,
            "source": source, "target": {"reasoning": "I rewrite the pad.", "content": "",
                                         "tool_call": {"name": "update_scratchpad", "arguments": {"contents": contents}}}}


def dropped(stats, name):
    return sum(stats["dropped"].get(name, {}).values())


# ---- 1 pad format ----

def test_action_after_hand_one_needs_a_v2_pad():
    old = record(messages(hand=5, pad_text="(empty)", hole="7h 2c"))
    free = record(messages(hand=5, pad_text="notes: seat0 loose", hole="7h 2c"))
    ok = record(messages(hand=5, pad_text=pad(5), hole="7h 2c"))
    first = record(messages(hand=1, pad_text="", hole="7h 2c"))
    out, stats = clean([old, free, ok, first])
    assert out == [ok, first]
    assert stats["dropped"]["no_v2_pad"] == {"self_clean": 2}


# ---- 2 pad_teacher ----

def test_pad_teacher_drops_restarted_counters_and_game_over():
    good = pad_record(10, pad(10))
    off_by_one = pad_record(10, pad(10, vb=9))
    restart = pad_record(10, pad(10, vb=3))
    over = pad_record(30, pad(30))
    late = pad_record(29, pad(29))
    out, stats = clean([good, off_by_one, restart, over, late])
    assert out == [good, off_by_one, late]
    assert dropped(stats, "pad_restart") == 1 and dropped(stats, "pad_game_over") == 1
    assert stats["dropped"]["pad_restart"] == {"pad_teacher": 1}


def test_pad_teacher_with_free_text_pad_is_kept_by_the_counter_check():
    free = pad_record(12, "my notes only")
    assert clean([free])[0] == [free]


# ---- 4 legality ----

def test_illegal_targets_are_dropped():
    m = messages(legal="fold, call 2, raise to 4-200")
    only_check_raise = messages(legal="check, raise to 4-200")
    check_bet = messages(legal="check, bet to 2-200", street="flop", board="2c 7d Ks", pot=6)
    call_only = messages(legal="fold, call 2", hole="Ah Ad")
    legal_ones = [record(m, "call", reasoning="call"), record(m, "raise", 10, reasoning="raise AA", source="teacher_agree"),
                  record(check_bet, "bet", 4, reasoning="bet"), record(only_check_raise, "check", reasoning="check"),
                  record(call_only, "call", reasoning="call"), record(m, "raise", 200, reasoning="shove")]
    bad = [record(only_check_raise, "call"), record(only_check_raise, "fold"), record(m, "raise"),
           record(m, "raise", 3), record(m, "raise", 201), record(check_bet, "bet", 1), record(check_bet, "raise", 6),
           record(call_only, "raise", 10), record(m, "check")]
    out, stats = clean(legal_ones + bad)
    assert out == legal_ones
    assert dropped(stats, "illegal") == len(bad)


# ---- 5 first-in folds ----

def test_first_in_fold_of_a_chart_hand_is_dropped():
    # hero seat 2 = UTG (position 2). UTG chart has AQo+, 44+.
    utg_aqo = record(messages(hole="Ah Qc", folded=(), legal="fold, call 2, raise to 4-200"), "fold")
    utg_44 = record(messages(hole="4h 4c"), "fold")
    utg_72 = record(messages(hole="7h 2c"), "fold")
    raised = record(messages(hole="Ah Qc", events=["preflop: seat4 (seat 4) raise to 6"], bets={0: 1, 1: 2, 4: 6},
                             legal="fold, call 6, raise to 10-200"), "fold")
    limped = record(messages(hole="Ah Qc", events=["preflop: seat4 (seat 4) call"], bets={0: 1, 1: 2, 4: 2}), "fold")
    raise_it = record(messages(hole="Ah Qc"), "raise", 5)
    out, stats = clean([utg_aqo, utg_44, utg_72, raised, limped, raise_it])
    assert out == [utg_72, raised, limped, raise_it]
    assert stats["dropped"]["rfi_fold"] == {"self_clean": 2}


def test_first_in_fold_is_kept_in_the_last_hands():
    # hand 28 of 30: folding a chart hand can be right when protecting a place
    late = record(messages(hand=28, hole="Ah Qc"), "fold")
    out, stats = clean([late])
    assert out == [late] and not stats["dropped"].get("rfi_fold")


def test_first_in_fold_after_folds_counts_and_position_comes_from_the_seat():
    # hero seat 5 = BTN (position 5, button seat 5), UTG/HJ/CO folded: 76s is a BTN open, a fold is dropped
    folds = ["preflop: seat2 (seat 2) fold", "preflop: seat4 (seat 4) fold"]
    btn = record(messages(seat=5, position=4, button=5, hole="7h 6h", events=folds, folded=(2, 4),
                          bets={0: 1, 1: 2}), "fold", source="teacher_resampled")
    out, stats = clean([btn])
    assert out == [] and stats["dropped"]["rfi_fold"] == {"teacher_resampled": 1}


def test_short_stack_first_in_fold_is_kept():
    m = messages(hole="Ah Qc")
    m[2]["content"] = m[2]["content"].replace("seat 2: seat2 (you) stack 200", "seat 2: seat2 (you) stack 20")
    shallow = record(m, "fold")
    assert clean([shallow])[0] == [shallow]


# ---- 6 bad folds ----

def test_bad_fold_with_a_made_hand_or_big_draw_and_a_cheap_price_is_dropped():
    flop = dict(street="flop")
    top_pair = messages(hole="Ah 9c", board="As 7d 2c", pot=20, legal="fold, call 4, raise to 8-190", **flop)
    top_pair_pricey = messages(hole="Ah 9c", board="As 7d 2c", pot=10, legal="fold, call 8, raise to 16-190", **flop)
    flush_draw = messages(hole="Ah 9h", board="Kh 7h 2c", pot=20, legal="fold, call 5, raise to 10-190", **flop)
    air = messages(hole="Ah 9c", board="Ks 7d 2c", pot=20, legal="fold, call 4, raise to 8-190", **flop)
    bad = [record(top_pair, "fold"), record(flush_draw, "fold")]
    keep = [record(top_pair_pricey, "fold"), record(air, "fold"), record(top_pair, "call")]
    out, stats = clean(bad + keep)
    assert out == keep and dropped(stats, "bad_fold") == 2


# ---- 7 leakage ----

def test_leakage_words_content_and_pasted_json_are_dropped():
    m = messages(hole="7h 2c")
    bad = [record(m, reasoning="The suggested action is fold."), record(m, reasoning="A hint says fold."),
           record(m, reasoning="The teacher folds here."), record(m, reasoning="From my training I fold."),
           record(m, reasoning="An LLM would fold."), record(m, reasoning="fold", content="Analysis: weak hand."),
           record(m, reasoning='I will call submit_action {"action": "fold"} now.'),
           pad_record(10, pad(10), source="pad_teacher") | {"target": {"reasoning": "The teacher wrote this.", "content": "",
                                                                         "tool_call": {"name": "update_scratchpad", "arguments": {"contents": pad(10)}}}}]
    keep = [record(m, reasoning="Fold: seven deuce is trash, and the whitelist stays; I hinge on price."),
            record(m, reasoning="fold", content="  \n")]
    out, stats = clean(bad + keep)
    assert out == keep and dropped(stats, "leakage") == len(bad)


# ---- 8 confusion ----

def test_three_waits_or_rereads_drop_an_action_record():
    m = messages(hole="7h 2c")
    bad = [record(m, reasoning="Wait, no. Wait, I mean fold. Wait again."),
           record(m, reasoning="Let me re-read. Re-read the table. Wait. fold"),
           record(m, reasoning="WAIT, no. wait let me see. Wait!")]
    keep = [record(m, reasoning="Wait, the button is seat 5. Fold."), record(m, reasoning="I wait for a better spot. Fold."),
            record(m, reasoning="I can wait for a better spot, wait to see more hands, and wait for a good pot. Fold.")]
    out, stats = clean(bad + keep)
    assert out == keep and dropped(stats, "confusion") == 3


# ---- 3 suitedness ----

def test_suitedness_contradiction_drops_action_records_but_not_pad_records():
    m = messages(hole="5c Qc", legal="fold, call 2, raise to 4-200", position=2, seat=2)
    bad = record(m, "call", reasoning="Hand strength: 5c Qc is offsuit and weak.", source="teacher_agree")
    ok = record(m, "call", reasoning="Hand strength: 5c Qc is suited, Q5s is fine in position.")
    pad_rec = pad_record(10, pad(10)) | {"target": {"reasoning": "5c Qc is offsuit.", "content": "",
                                                     "tool_call": {"name": "update_scratchpad", "arguments": {"contents": pad(10)}}}}
    out, stats = clean([bad, ok, pad_rec])
    assert out == [ok, pad_rec] and stats["dropped"]["suitedness"] == {"teacher_agree": 1}


# ---- stats ----

def test_stats_report_input_kept_and_first_filter_that_drops():
    m = messages(hole="7h 2c")
    r = record(m, "call", reasoning="The teacher says wait wait wait.")  # illegal? no: legal call; leakage first
    out, stats = clean([r, record(m, reasoning="fold")])
    assert stats["in"] == 2 and stats["out"] == 1 and stats["dropped_total"] == 1
    assert dropped(stats, "leakage") == 1 and dropped(stats, "confusion") == 0


# ---- pad read line ----

def test_pad_read_line_keeps_survive_and_only_relevant_seat_lines():
    # UTG hero, nobody has acted: no steal line; seat lines only for seats still in the hand (0 SB, 1 BB, 4, 5; not 3)
    m = messages(hole="7h 2c")
    out, stats = add_pad_read_line([record(m, reasoning="Fold.")])
    head = out[0]["target"]["reasoning"].split("\n")[0]
    assert head.startswith("Reads: > ") is False
    assert head == ("Reads: survive: leading late, avoid big pots with marginal hands; "
                    "seat0 folds to raises (5/5): raise first-in and 3-bet light; seat1 is tight (VPIP 1/10): respect its raises; "
                    "seat5 calls a lot: value bet, do not bluff")
    assert out[0]["target"]["reasoning"].endswith("\nFold.") and stats["with_pad"] == 1


def test_pad_read_line_skips_every_folded_seat():
    # seats 4 and 5 folded (4 with a shown fold action, 5 without): neither can affect the decision
    pad_text = pad(10, lines=["> seat4 is tight (VPIP 1/10): respect its raises", "> seat5 calls a lot: value bet, do not bluff"])
    m = messages(hole="7h 2c", folded=(4, 5), events=["preflop: seat4 (seat 4) fold"], pad_text=pad_text)
    out, _ = add_pad_read_line([record(m, reasoning="Fold.")])
    assert out[0]["target"]["reasoning"].split("\n")[0] == "Reads: nothing relevant"


def test_pad_read_line_steal_line_only_when_first_in_from_a_late_seat():
    # ring 0,1,2,4,5 with button 2: SB 4, BB 5, UTG 0, CO 1 (position 3), BTN 2. Hero seat 1 = CO, first in.
    steal = pad(10, lines=["> blinds fold to steals (5/5): open BTN/CO wide"])
    folds = ["preflop: seat0 (seat 0) fold"]
    co_first = messages(seat=1, position=3, button=2, hole="7h 2c", events=folds, folded=(0,), bets={4: 1, 5: 2}, pad_text=steal)
    out, _ = add_pad_read_line([record(co_first, reasoning="Fold.")])
    assert out[0]["target"]["reasoning"].startswith("Reads: blinds fold to steals (5/5): open BTN/CO wide\n")
    limped = messages(seat=1, position=3, button=2, hole="7h 2c", events=folds + ["preflop: seat2 (seat 2) call"], folded=(0,),
                      bets={4: 1, 5: 2, 2: 2}, pad_text=steal)
    out, _ = add_pad_read_line([record(limped, reasoning="Fold.")])
    assert out[0]["target"]["reasoning"].startswith("Reads: nothing relevant\n")
    utg = messages(seat=0, position=2, button=2, hole="7h 2c", bets={4: 1, 5: 2}, pad_text=steal)  # not a late seat
    out, _ = add_pad_read_line([record(utg, reasoning="Fold.")])
    assert out[0]["target"]["reasoning"].startswith("Reads: nothing relevant\n")


def test_pad_read_line_skips_records_without_a_v2_pad_and_pad_records_and_is_idempotent():
    free = record(messages(pad_text="(empty)"), reasoning="Fold.")
    pad_rec = pad_record(10, pad(10))
    out, stats = add_pad_read_line([free, pad_rec])
    assert out == [free, pad_rec] and stats["no_pad"] == 1 and stats["not_action"] == 1
    once, _ = add_pad_read_line([record(messages(), reasoning="Fold.")])
    twice, stats = add_pad_read_line(once)
    assert twice == once and stats["already"] == 1


def test_pad_read_line_goes_below_a_facts_line_and_fact_header_goes_above_it():
    from poker.sft.make_records import add_fact_headers

    rec = record(messages(hole="7h 2c"), reasoning="Fold.")
    padded, _ = add_pad_read_line([rec])
    headed, _ = add_fact_headers(padded)
    lines = headed[0]["target"]["reasoning"].split("\n")
    assert lines[0].startswith("Facts: ") and lines[1].startswith("Reads: ") and lines[2] == "Fold."
    # the other order lands in the same place, and neither step repeats itself
    other, _ = add_pad_read_line(add_fact_headers([rec])[0])
    assert other[0]["target"]["reasoning"] == headed[0]["target"]["reasoning"]
    assert add_fact_headers(headed)[0] == headed and add_pad_read_line(headed)[0] == headed


def test_clean_leaves_input_unchanged_and_records_are_json_serialisable():
    m = messages(hole="7h 2c")
    records = [record(m, reasoning="Fold.")]
    before = json.dumps(records)
    out, _ = clean(records)
    assert json.dumps(records) == before and json.dumps(out)


# ---- c8 target filters ----

def _facts_free(reasoning, action="fold", legal="fold, call 2, raise to 4-200", amount=None, **kw):
    return record(messages(hand=5, pad_text=pad(5), legal=legal), action=action, amount=amount, reasoning=reasoning, **kw)


BODY = ("Seat 0 raised and I hold a weak offsuit hand out of position. The price is 40% of the pot, so I need real equity, "
        "and this hand has little against a raiser. Folding costs nothing more. ")


def test_legal_target_checks_verbs_and_ranges():
    from poker.sft.clean import legal_target

    assert legal_target(_facts_free(BODY, "call"))
    assert legal_target(_facts_free(BODY, "raise", amount=4))
    assert legal_target(_facts_free(BODY, "raise", amount=200))
    assert not legal_target(_facts_free(BODY, "raise", amount=3))
    assert not legal_target(_facts_free(BODY, "raise", amount=201))
    assert not legal_target(_facts_free(BODY, "raise"))  # no amount
    assert not legal_target(_facts_free(BODY, "bet", amount=10))  # bet not legal here
    assert not legal_target(_facts_free(BODY, "check"))
    assert not legal_target(_facts_free(BODY, "allin"))
    assert not legal_target(_facts_free(BODY, "fold", legal="check, bet to 2-100"))
    assert legal_target(pad_record(3, "x"))  # not an action turn


def test_illegal_filter_rejects_out_of_range_and_unknown_verbs():
    bad = [_facts_free(BODY, "raise", amount=3), _facts_free(BODY, "raise", amount=999), _facts_free(BODY, "allin"),
           _facts_free(BODY, "raise")]
    out, stats = clean(bad)
    assert out == [] and dropped(stats, "illegal") == 4


def test_strict_filters_are_opt_in():
    r = _facts_free("Fold.")
    assert clean([r])[0] == [r]
    out, stats = clean([r], strict=True)
    assert out == [] and dropped(stats, "short_body") == 1


def test_short_and_long_body_ignore_header_lines():
    head = "Facts: pos 2 of 5 (0=SB), x\nReads: nothing relevant\nTalk: none\nDecision: fold\n"
    short = _facts_free(head + "Fold, weak.")
    long = _facts_free(BODY * 20)
    ok = _facts_free("Facts: pos 2 of 5\n" + BODY)
    out, stats = clean([short, long, ok], strict=True)
    assert out == [ok]
    assert dropped(stats, "short_body") == 1 and dropped(stats, "long_body") == 1


def test_tool_text_filter():
    texts = ["<tool_call><function=submit_action>", "then <function=submit_action>", "</tool_response>", '{"name": "submit_action"}']
    recs = [_facts_free(BODY + t) for t in texts] + [_facts_free(BODY, content="ok <tool_call>")]
    out, stats = clean(recs, strict=True)
    assert out == [] and dropped(stats, "tool_text") == 5


def test_conclusion_mismatch_filter():
    mismatched = [
        _facts_free(BODY + "I should fold.", "call"),
        _facts_free(BODY + "So I'll fold here.", "raise", amount=10),
        _facts_free(BODY + "Folding is best.", "call"),
        _facts_free(BODY + "I'll call.", "fold"),
        _facts_free(BODY + "I'll raise.", "call"),
    ]
    fine = [
        _facts_free(BODY + "I'll fold.", "fold"),
        _facts_free(BODY + "I'll call.", "call"),
        _facts_free(BODY + "I could fold, but I'll call.", "call"),  # both verbs
        _facts_free(BODY + "I'll fold to a reraise, but I'll call now.", "call"),
        _facts_free(BODY + "I shouldn't fold. Call.", "call"),
        _facts_free(BODY + "I'll bet.", "raise", amount=10),  # same family
        _facts_free("I should fold " + "x" * 400 + BODY + "I'll call.", "call"),  # old text outside the last 250 chars
    ]
    out, stats = clean(mismatched + fine, strict=True)
    assert out == fine, [r["target"]["reasoning"][-40:] for r in out]
    assert dropped(stats, "conclusion_mismatch") == 5


def test_no_say_filter_and_strip():
    from poker.sft.clean import strip_say

    r = _facts_free(BODY)
    r["target"]["tool_call"]["arguments"]["say"] = "nice"
    assert clean([r], strict=True)[0] == [r]  # opt-in
    out, stats = clean([r], strict=True, no_say=True)
    assert out == [] and dropped(stats, "has_say") == 1
    stripped, n = strip_say([r])
    assert "say" not in stripped[0]["target"]["tool_call"]["arguments"] and n == 1
    assert r["target"]["tool_call"]["arguments"]["say"] == "nice"  # input not mutated


# ---- decision line ----

def test_decision_line_variants_and_idempotence():
    from poker.sft.clean import add_decision_line

    recs = [_facts_free(BODY, "fold"), _facts_free(BODY, "call"), _facts_free(BODY, "raise", amount=12),
            record(messages(hand=5, pad_text=pad(5), legal="check, bet to 2-200"), action="bet", amount=7, reasoning=BODY),
            record(messages(hand=5, pad_text=pad(5), legal="check, bet to 2-200"), action="check", reasoning=BODY + "\n")]
    out, stats = add_decision_line(recs)
    last = [r["target"]["reasoning"].split("\n")[-1] for r in out]
    assert last == ["Decision: fold", "Decision: call 2", "Decision: raise to 12", "Decision: bet 7", "Decision: check"]
    assert stats["added"] == 5
    again, stats = add_decision_line(out)
    assert again == out and stats["added"] == 0 and stats["already"] == 5
    assert recs[0]["target"]["reasoning"] == BODY


def test_decision_line_skips_pad_turns_and_unbuildable():
    from poker.sft.clean import add_decision_line

    pad_r = pad_record(3, "x")
    illegal = _facts_free(BODY, "raise")  # no amount
    out, stats = add_decision_line([pad_r, illegal])
    assert out == [pad_r, illegal] and stats["not_action"] == 1 and stats["unbuildable"] == 1


def test_decision_line_ignores_prose_that_starts_with_decision():
    from poker.sft.clean import add_decision_line

    prose = _facts_free(BODY + "\nDecision: Fold 7c Jc here.", "fold")
    out, stats = add_decision_line([prose])
    assert out[0]["target"]["reasoning"].endswith("Decision: Fold 7c Jc here.\nDecision: fold") and stats["added"] == 1


def test_no_say_alone_does_not_switch_on_the_strict_filters():
    m = messages(hole="7h 2c")
    short = record(m, reasoning="Fold.")  # under MIN_BODY: only --clean-strict drops it
    out, stats = clean([short], no_say=True)
    assert out == [short] and not stats["dropped"]


def test_strict_drops_non_ascii_and_think_tags():
    m = messages(hole="7h 2c")
    body = "This hand is weak from early position and there is nothing to gain by playing it against five players. " * 2
    ok = record(m, reasoning=body + "Fold.")
    cjk = record(m, reasoning=body + "弃牌是标准操作。Fold.")
    tag = record(m, reasoning=body + "</thinking> Fold.")
    suits = record(m, reasoning=body + "I hold 7\u2665 2\u2663 \u2014 a fold.")
    out, stats = clean([ok, cjk, tag, suits], strict=True)
    assert out == [ok, suits] and dropped(stats, "junk_text") == 2


def test_strip_model_headers_removes_the_models_own_header_lines():
    from poker.sft.clean import strip_model_headers

    m = messages(hole="7h 2c")
    r = record(m, reasoning="Facts: UTG (9 behind), 72s.\nReads: nothing relevant\nFacts: BB, 72o.\nTalk: seat3 is noise.\n"
                            "This hand is weak.\nI fold.\nDecision: fold")
    pad_rec = pad_record(10, pad(10))
    out, stats = strip_model_headers([r, pad_rec])
    assert out[0]["target"]["reasoning"] == "This hand is weak.\nI fold."
    assert out[1] == pad_rec and stats == {"stripped": 1}
    assert r["target"]["reasoning"].startswith("Facts: UTG")  # input not mutated
