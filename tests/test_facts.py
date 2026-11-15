import pytest

from poker.facts import contradictions, fact_line, compute_facts, render_facts

SYSTEM = {"role": "system", "content": "rules"}


def prompt(*, hand=17, total=30, seat=4, position=0, street="preflop", hole="Qh Tc", board="-", pot=9,
           table=None, legal="fold, call 5, raise to 10-188", since=None):
    """A prompt shaped like the arena's: first message of the hand, optionally a later 'since last action' one."""
    table = table or ["  seat 0: seat0 stack 191, bet 0, folded", "  seat 1: seat1 stack 180, bet 0, folded",
                      "  seat 2: seat2 stack 216, bet 6", "  seat 3: seat3 [button] stack 337, bet 0, folded",
                      "  seat 4: seat4 (you) stack 187, bet 1", "  seat 5: seat5 stack 80, bet 2"]
    head = f"## Hand {hand} of {total}\nYour seat: {seat} (position {position}; 0 = small blind)\n"
    state = (f"Street: {street}\nYour hole cards: {hole}\nBoard: {board}\nPot (including current bets): {pot}\n"
             "Table:\n" + "\n".join(table) + f"\n\nLegal actions: {legal}\nCall submit_action with your decision.")
    if since is None:
        return [SYSTEM, {"role": "user", "content": head + state}]
    return [SYSTEM, {"role": "user", "content": head + "Street: preflop\nYour hole cards: " + hole + "\nBoard: -\nPot (including current bets): 3\nTable:\n  seat 0: x stack 1, bet 0\n\nLegal actions: check\n"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": f"Since your last action:\n  {since}\n\n" + state}]


def ring(stacks, button, hero, folded=(), out=(), bets=None):
    bets = bets or {}
    lines = []
    for s, st in enumerate(stacks):
        tag = " [button]" if s == button else ""
        you = " (you)" if s == hero else ""
        status = ", out" if s in out else (", folded" if s in folded else "")
        lines.append(f"  seat {s}: seat{s}{tag}{you} stack {st}, bet {bets.get(s, 0)}{status}")
    return lines


def test_preflop_btn_facts_and_line():
    # button seat 3; SB seat 4, BB seat 5, UTG 0, HJ 1, CO 2, BTN 3. Hero on the button with AJs.
    table = ring([200, 200, 200, 210, 200, 190], button=3, hero=3, bets={4: 1, 5: 2})
    msgs = prompt(seat=3, position=5, hole="Ah Jh", pot=3, table=table, legal="fold, call 2, raise to 4-210", hand=12)
    f = compute_facts(msgs)
    assert f["position"] == "BTN" and f["behind"] == 2 and f["hole_code"] == "AJs"
    assert f["to_call"] == 2 and f["required"] == pytest.approx(0.4)
    assert f["hands_left"] == 18 and f["rank"] == 1 and f["n_live"] == 6
    line = render_facts(f)
    assert line.startswith("Facts: BTN (2 behind), AJs, board -, no made hand, pot 3, call 2 (40%)")
    assert "stack 105 BB" in line and "rank 1/6" in line and "18 hands left" in line
    assert fact_line(msgs) == line


def test_position_names_full_ring():
    table = ring([200] * 6, button=3, hero=0)
    names = {}
    for seat, pos in [(0, 2), (1, 3), (2, 4), (3, 5), (4, 0), (5, 1)]:
        names[seat] = compute_facts(prompt(seat=seat, position=pos, table=ring([200] * 6, 3, seat)))["position"]
    assert names == {0: "UTG", 1: "HJ", 2: "CO", 3: "BTN", 4: "SB", 5: "BB"}


def test_short_handed_names_and_busted_seats_are_skipped():
    # seat 3 busted; live ring 0,1,2,4,5; button seat 2 -> SB 4, BB 5, UTG 0, CO 1, BTN 2
    def pos(seat, p):
        return compute_facts(prompt(seat=seat, position=p, hole="Kh 7s", table=ring([213, 191, 386, 0, 197, 198], 2, seat, out=(3,))))
    assert pos(0, 2)["position"] == "UTG" and pos(1, 3)["position"] == "CO" and pos(2, 4)["position"] == "BTN"
    assert pos(4, 0)["position"] == "SB" and pos(5, 1)["position"] == "BB"
    assert pos(5, 1)["n_live"] == 5


def test_heads_up():
    table = ring([300, 100], button=0, hero=0)
    f = compute_facts(prompt(seat=0, position=0, hole="9c 9d", table=table, pot=3))
    assert f["position"] == "SB/BTN" and f["n_live"] == 2


def test_behind_counts_only_live_players_that_have_not_folded():
    # hero UTG (seat 0): everyone behind still in -> 5; UTG in 5-handed, seat 1 folded already can't be (UTG acts first)
    table = ring([200] * 6, button=3, hero=0)
    assert compute_facts(prompt(seat=0, position=2, table=table))["behind"] == 5
    # hero CO (seat 2); seat 3 (BTN) and 4 folded, BB (5) live => behind = 1 (BB); SB folded
    table = ring([200] * 6, button=3, hero=2, folded=(0, 1, 3, 4), bets={5: 2})
    assert compute_facts(prompt(seat=2, position=4, table=table))["behind"] == 1


def test_postflop_behind_uses_postflop_order_and_made_hand():
    # flop, hero SB (pos 0) acts first: all live others behind
    table = ring([200] * 6, button=3, hero=4, folded=(0, 1))
    msgs = prompt(seat=4, position=0, street="flop", hole="Ah Kd", board="Ks 7d 2c", pot=12, table=table,
                  legal="check, bet to 2-198", since="(no other actions)")
    f = compute_facts(msgs)
    assert f["behind"] == 3 and f["made"] == "top pair" and f["draw"] is None
    assert "top pair" in render_facts(f)


@pytest.mark.parametrize("hole,board,made", [
    ("As Ah", "Ks 7d 2s", "overpair"), ("7c 7d", "7s Kd 2s", "set"), ("2c 3d", "7s Kd 2s", "bottom pair"),
    ("Kc 3d", "7s Kd 2s", "top pair"), ("Qc 7d", "7s Kd 2s", "second pair"), ("Ac Qd", "7s Kd 2s", None),
    ("Kc Qd", "7s 7d 2s", "board pair only"),  # the board pair is not ours
    ("7c 2d", "7s 7d 2s", "full house"), ("Ac Ks", "As Kd 2s", "two pair"), ("Ah 5h", "Kh 9h 2h", "flush"),
    ("Jc Td", "9s 8d 7c", "straight"), ("Kc Jd", "Kd 7c 7h", "two pair"), ("Qc Td", "7d 7c 7h", "playing the board"),
    ("8c 8d", "8h 8s 2c", "quads"), ("Ac Kc", "Qs Jd Th", "straight"), ("9c 9d", "Kh 5s 2s", "pocket pair"),
    ("Ah 5h", "Kc 5c 2d", "second pair"),
])
def test_made_hand_names(hole, board, made):
    f = compute_facts(prompt(street={3: "flop", 4: "turn", 5: "river"}[len(board.split())], hole=hole, board=board,
                             legal="check, bet to 2-100", pot=20))
    assert f["made"] == made


def test_preflop_pocket_pair_is_made_pair_and_unsuited_codes():
    assert compute_facts(prompt(hole="9c 9d"))["made"] == "pocket pair"
    assert compute_facts(prompt(hole="9c 9d"))["hole_code"] == "99"
    assert compute_facts(prompt(hole="Qh Tc"))["hole_code"] == "QTo"
    assert compute_facts(prompt(hole="Qh Th"))["hole_code"] == "QTs"


@pytest.mark.parametrize("hole,board,outs,kind", [
    ("Ah 5h", "Kh 9h 2c", 9, "flush draw"), ("Jc Td", "9s 8d 2c", 8, "open-ended"), ("Jc Td", "9s 7d 2c", 4, "gutshot"),
    ("Jh Th", "9h 8h 2c", 15, "flush draw"),         # flush draw plus open-ended: 9 + 8 - 2 shared
    ("Ac Kd", "7s 7d 2c", 0, None),
])
def test_draws_and_outs(hole, board, outs, kind):
    f = compute_facts(prompt(street="flop", hole=hole, board=board, legal="check, bet to 2-100", pot=20))
    assert (f["draw"] or {}).get("outs", 0) == outs
    if kind:
        assert kind in f["draw"]["name"]


def test_no_draw_on_river_or_when_made():
    f = compute_facts(prompt(street="river", hole="Ah 5h", board="Kh 9h 2c 3d 8s", legal="check, bet to 2-100"))
    assert f["draw"] is None
    f = compute_facts(prompt(street="flop", hole="Ah 5h", board="Kh 9h 2h", legal="check, bet to 2-100"))
    assert f["made"] == "flush" and f["draw"] is None


def test_rank_gaps_and_ties():
    stacks = [250, 260, 200, 300, 120, 100]  # hero seat 2 chips 200: 300, 260, 250 above => rank 4
    f = compute_facts(prompt(seat=2, position=4, table=ring(stacks, 3, 2)))
    assert f["rank"] == 4 and f["gap_up"] == 50 and f["gap_down"] == 80
    assert "rank 4/6 (+50 to 3rd, -80 to 5th)" in render_facts(f)
    f = compute_facts(prompt(seat=3, position=5, table=ring(stacks, 3, 3)))
    assert f["rank"] == 1 and f["gap_up"] is None
    assert "rank 1/6 (-40 to 2nd)" in render_facts(f)
    last = compute_facts(prompt(seat=5, position=1, table=ring(stacks, 3, 5)))
    assert last["rank"] == 6 and last["gap_down"] is None and "rank 6/6 (+20 to 5th)" in render_facts(last)


def test_rank_counts_chips_in_front_and_only_live_players():
    table = ring([200, 200, 100, 0, 150, 150], button=2, hero=4, out=(3,), bets={4: 1, 5: 2})
    f = compute_facts(prompt(seat=4, position=0, table=table, pot=3))
    assert f["n_live"] == 5 and f["chips"] == 151


def test_check_spot_and_unparseable():
    f = compute_facts(prompt(street="flop", hole="Ac Qd", board="7s Kd 2s", pot=7, legal="check, bet to 2-198"))
    assert f["to_call"] == 0 and "call" not in render_facts(f).split("pot 7")[1] and "no bet" in render_facts(f)
    assert compute_facts([{"role": "user", "content": "hello"}]) is None
    assert fact_line([{"role": "user", "content": "hello"}]) is None


# ---- hallucination checks -------------------------------------------------------------------------------------

def facts_for(hole, board="-", position="BTN", **kw):
    street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[0 if board == "-" else len(board.split())]
    legal = "fold, call 2, raise to 4-200" if street == "preflop" else "check, bet to 2-100"
    seat, p = {"UTG": (0, 2), "HJ": (1, 3), "CO": (2, 4), "BTN": (3, 5), "SB": (4, 0), "BB": (5, 1)}[position]
    return compute_facts(prompt(seat=seat, position=p, street=street, hole=hole, board=board, legal=legal,
                                table=ring([200] * 6, 3, seat), pot=3 if street == "preflop" else 20))


@pytest.mark.parametrize("text", [
    "I'm on the button, so I can open wide.",
    "I am in the BTN here.", "I'm sitting in the button and acting last.", "My position is BTN.",
])
def test_position_claim_wrong_for_utg(text):
    assert any("position" in c for c in contradictions(text, facts_for("Ah Jh", position="UTG")))


def test_position_claims_that_agree_or_are_not_first_person_pass():
    f = facts_for("Ah Jh", position="UTG")
    for text in ["I'm UTG, first to act.", "I'm facing a raise from the button.", "The button raised and I am UTG.",
                 "I'm under the gun with AJs.", "The player on the button is aggressive; I have AJs."]:
        assert not [c for c in contradictions(text, f) if "position" in c], text
    f = facts_for("Ah Jh", position="BTN")
    assert not contradictions("I'm on the button with AJs, I should open.", f)
    assert contradictions("I'm in early position, so I should be careful.", f)
    assert contradictions("I'm in late position.", facts_for("Ah Jh", position="UTG"))
    assert not contradictions("I'm in late position.", f)


def test_position_claim_heads_up_and_big_blind():
    assert contradictions("I'm on the button.", facts_for("Ah Jh", position="BB"))
    assert not contradictions("I'm in the big blind.", facts_for("Ah Jh", position="BB"))


@pytest.mark.parametrize("text,hole,board", [
    ("I have a set of sevens here.", "Ac Kd", "7s Kh 2c"),
    ("I flopped a flush, so I bet.", "Ac Kd", "7s Kh 2c"),
    ("I have a straight now.", "Ac Kd", "7s 8d 2c"),
    ("I have two pair, kings and sevens.", "Ac Kd", "7s Kh 2c"),
    ("I have top pair with a good kicker.", "Ac Qd", "7s Kh 2c"),
])
def test_hand_claims_that_the_cards_do_not_make(text, hole, board):
    f = facts_for(hole, board)
    assert contradictions(text, f), text


@pytest.mark.parametrize("text,hole,board", [
    ("I have a set of sevens.", "7c 7d", "7s Kh 2c"), ("I flopped trips.", "7c Ad", "7s 7h 2c"),
    ("I have a flush draw with 9 outs.", "Ah 5h", "Kh 9h 2c"), ("I have the nut flush draw, 9 outs.", "Ah 5h", "Kh 9h 2c"),
    ("I have an open-ended straight draw.", "Jc Td", "9s 8d 2c"), ("If I have a set I'm happy, but I don't.", "Ac Kd", "7s Kh 2c"),
    ("I don't have a flush or a straight. I only have top pair.", "Ac Kd", "7s Kh 2c"),
    ("I have a pair of kings.", "Ac Kd", "7s Kh 2c"), ("I have top pair.", "Kc Jd", "Ks 7h Td"), ("I have overpair aces.", "Ac Ad", "7s Kh 2c"),
    ("I have two pair.", "Ac Kd", "As Kh 2c"), ("I hit my flush.", "Ah 5h", "Kh 9h 2h"),
    ("I made a straight.", "Jc Td", "9s 8d 7c"), ("I have a full house, a set plus board pair.", "7c 7d", "7s Kd Kc"),
])
def test_correct_claims_are_not_flagged(text, hole, board):
    assert contradictions(text, facts_for(hole, board)) == []


def test_flush_and_straight_draw_text_is_not_a_made_claim():
    f = facts_for("Ac Kd", "7s 8d 2c")
    assert contradictions("I have a flush draw? No. I only have ace high, a straight draw is not there.", f) == []
    assert contradictions("I'm hoping to hit a flush or a straight.", f) == []


def test_pair_suited_and_suitedness_claims():
    f = facts_for("7c 7d")
    assert any("suited" in c for c in contradictions("I have 77s, a small pair suited.", f))
    assert any("suited" in c for c in contradictions("I have AJs here.", facts_for("Ah Jc")))
    assert not contradictions("I have AJs here.", facts_for("Ah Jh"))
    assert not contradictions("I have AJo here.", facts_for("Ah Jc"))
    assert not contradictions("An opponent could hold AJs.", facts_for("Ah Jc"))


def test_wrong_outs_by_more_than_two():
    f = facts_for("Ah 5h", "Kh 9h 2c")  # flush draw, 9 outs
    assert any("outs" in c for c in contradictions("I have a flush draw with 12 outs.", f))
    assert any("outs" in c for c in contradictions("My flush draw has 4 outs.", f))
    assert not contradictions("I have a flush draw with 9 outs.", f)
    assert not contradictions("I have a flush draw with 8 outs, about 35% by the rule of four.", f)
    assert not contradictions("I have a flush draw with 10 outs.", f)
    f = facts_for("Jc Td", "9s 8d 2c")  # OESD 8 outs
    assert any("outs" in c for c in contradictions("I have an open-ended straight draw, 15 outs.", f))
    assert not contradictions("My straight draw has 8 outs.", f)
    # a pair or overcard out count is not a draw claim
    assert not contradictions("I have 6 outs to hit an overcard pair.", f)
    assert not contradictions("The opponent has a flush draw with 12 outs.", f)


def test_no_contradictions_for_empty_text_or_missing_facts():
    assert contradictions("", facts_for("Ah Jh")) == []
    assert contradictions("I have a set.", None) == []


def test_only_the_last_stated_position_counts_and_past_hands_do_not():
    f = facts_for("Ah Jh", position="BTN")
    assert not contradictions("I'm in the small blind... wait, let me re-read. I'm on the button.", f)
    assert contradictions("I'm on the button... wait, no. I'm in the big blind.", f)
    assert not contradictions("Last hand I had K7o in the small blind and folded.", f)
    assert not contradictions("I have JAs (offsuit), so be careful.", facts_for("Ah Jc"))


def _river(hole, board):
    return compute_facts(prompt(street="river", hole=hole, board=board, legal="check, bet to 2-100", pot=20))


def test_made_hand_when_hero_category_equals_the_boards():
    assert _river("Ah 2c", "Kh Qh 9h 5h 3h")["made"] == "flush"          # the nut flush improves the board flush
    assert _river("2c 2d", "Kh Qh 9h 5h 3h")["made"] == "playing the board"
    assert _river("Ac Kd", "9s 8d 7c 6h 5h")["made"] == "playing the board"  # board straight, ours is no better
    assert _river("Tc 2d", "9s 8d 7c 6h 5h")["made"] == "straight"       # ten-high beats the board's nine-high
    assert _river("Kc 2d", "7s 7d 7c 9h 9s")["made"] == "playing the board"  # board full house
    assert _river("9c 2d", "9d 7d 7c 7h 2s")["made"] == "full house"
    assert _river("Ac Qd", "7s 7d 2c 9h 4s")["made"] == "board pair only"
    assert _river("Ac 7d", "7s 7h 2c 9h 4s")["made"] == "trips"
    assert _river("Ac Ad", "7s 7h 2c 9h 4s")["made"] == "two pair"


def test_made_hand_equal_category_claims():
    f = _river("2c 2d", "Kh Qh 9h 5h 3h")
    assert contradictions("I have a flush.", f) == [] or f["ambiguous"]
    f = _river("Ac Qd", "7s 7d 2c 9h 4s")
    assert any("hand" in c for c in contradictions("I have a pair of sevens.", f))


def test_heads_up_postflop_button_acts_last():
    table = ring([200, 200], button=0, hero=0)
    pre = compute_facts(prompt(seat=0, position=0, hole="9c 9d", table=table, pot=3))
    assert pre["behind"] == 1                     # SB/BTN acts first preflop, the BB is behind
    flop = dict(street="flop", board="Ks 7d 2c", legal="check, bet to 2-100", pot=4)
    assert compute_facts(prompt(seat=0, position=0, hole="9c 9d", table=table, **flop))["behind"] == 0   # SB/BTN acts last
    table = ring([200, 200], button=0, hero=1)
    assert compute_facts(prompt(seat=1, position=1, hole="9c 9d", table=table, **flop))["behind"] == 1   # BB acts first


def test_last_claim_per_kind_and_self_correction_markers():
    f = facts_for("Ac Kd", board="Ks 7d 2c")  # top pair
    assert not contradictions("I have a set. On reflection I have top pair.", f)       # last made-hand claim is true
    assert contradictions("I have top pair. I have a set.", f)                         # last one is false
    assert not contradictions("I have a set, wait, I actually have top pair.", f)      # marker sentence is skipped
    assert not contradictions("I have a flush draw with 15 outs. Recounting, I have a flush draw with 9 outs.", facts_for("Ah 5h", board="Kh 9h 2c"))
    g = facts_for("Ah Jc")
    assert not contradictions("I have AJs. Recounting: I have AJo.", g)
    assert contradictions("I have AJo. I have AJs.", g)
    # different kinds do not overwrite each other
    assert contradictions("I have a set. I'm on the button.", f)


# ---- suitedness claims about our own cards ----

def _suit_facts(hole):
    return compute_facts(prompt(hole=hole))


@pytest.mark.parametrize("hole,text", [
    ("5c Qc", "Hand strength: 5c Qc is offsuit, no pair, no playability."),
    ("Qc 5c", "5c Qc is offsuit"),
    ("Ah 2h", "I have ace-rag offsuit, which is weak."),
    ("Ah 2h", "Ace-rag offsuit is a fold here."),
    ("Kd Qs", "KQ suited is a fine hand to raise."),
    ("Kd Qs", "I hold KQ suited."),
    ("Kd Qs", "My hand is suited and connected."),
    ("Ah 2h", "My cards are offsuit."),
    ("Ah 2h", "I have an offsuit hand."),
    ("Ah 2h", "A2 off-suit is weak."),
    ("Ah 2h", "A2 unsuited is weak."),
])
def test_suitedness_claim_that_the_cards_disprove_is_flagged(hole, text):
    found = contradictions(text, _suit_facts(hole))
    assert any(c.startswith("suited") for c in found), found


@pytest.mark.parametrize("hole,text", [
    ("5c Qc", "5c Qc is suited, so Q5s has flush potential."),
    ("Ah 2h", "Ace-rag suited, a playable bluff-catcher."),
    ("Kd Qs", "KQ offsuit is fine to open."),
    ("Kd Qs", "Seat2 has KQ suited, I think."),            # about an opponent
    ("Kd Qs", "The opponent shows KQ suited at showdown."),
    ("Ah 2h", "The board is two-tone, so a flush draw is possible."),
    ("Ah 2h", "If A2 were offsuit I would fold."),         # hypothetical
    ("Kd Qs", "Wait, 5c Qc is suited? No, KQ offsuit."),   # self-correction sentence skipped
    ("Ah 2h", "Suited connectors play well against a caller."),   # generic talk, not our cards
    ("Ah Ad", "Pocket aces offsuit is the only way to hold them."),  # pocket pair: never flagged here
    ("Ah 2h", "A2s, suited ace, wheel potential."),
    ("Ah 2h", "I raise because the table folds."),
])
def test_suitedness_talk_that_is_right_or_not_about_our_cards_is_not_flagged(hole, text):
    found = contradictions(text, _suit_facts(hole))
    assert not any(c.startswith("suited") for c in found), found


def test_only_the_last_suitedness_claim_counts():
    facts = _suit_facts("5c Qc")
    assert not any(c.startswith("suited") for c in contradictions("5c Qc is offsuit. Hmm. 5c Qc is suited.", facts))


# ---- facts v2 ----

def test_facts_v2_line_copies_prompt_fields():
    from poker.facts import fact_line_v2, render_facts_v2

    table = ring([200, 200, 200, 210, 200, 190], button=3, hero=3, bets={4: 1, 5: 2})
    msgs = prompt(seat=3, position=5, hole="Ah Jh", pot=3, table=table, legal="fold, call 2, raise to 4-210", hand=12)
    line = fact_line_v2(msgs)
    assert line == ("Facts: pos 5 of 6 (0=SB, 1=BB), Ah Jh (AJs), board -, no made hand, pot 3, call 2 (40%), "
                    "legal fold | call 2 | raise to 4-210, stack 105 BB, hand 12/30.")
    assert render_facts_v2(compute_facts(msgs)) == line
    assert "BTN" not in line and "rank" not in line and "behind" not in line


def test_facts_v2_no_bet_and_board_and_dead_seats():
    from poker.facts import fact_line_v2

    table = ring([213, 191, 386, 0, 197, 198], 2, 4, out=(3,))
    msgs = prompt(seat=4, position=0, street="flop", hole="Ah Kd", board="Ks 7d 2c", pot=12, table=table,
                  legal="check, bet to 2-197", hand=3)
    line = fact_line_v2(msgs)
    assert line.startswith("Facts: pos 0 of 5 (0=SB, 1=BB), Ah Kd (AKo), board Ks 7d 2c, top pair, pot 12, no bet, legal check | bet to 2-197, ")
    assert line.endswith("stack 98 BB, hand 3/30.")


def test_facts_v2_without_hand_header_omits_hand():
    import re
    from poker.facts import fact_line_v2

    msgs = prompt(hand=5)
    msgs[1] = dict(msgs[1], content=re.sub(r"^## Hand .*\n", "", msgs[1]["content"]))
    assert "hand " not in fact_line_v2(msgs).split("stack")[1]


def test_contradictions_skips_position_when_v1_name_absent():
    f = {"hole": ["Ah", "Jh"], "hole_code": "AJs", "street": "preflop", "made": None, "ambiguous": False, "draw": None}
    assert contradictions("I'm in the SB here.", f) == []
    g = f | {"position": "BTN"}
    assert contradictions("I'm in the SB here.", g) == ["position: says SB but we are BTN"]
