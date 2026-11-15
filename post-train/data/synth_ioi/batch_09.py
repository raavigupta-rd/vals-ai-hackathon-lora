"""Batch 9 (synth_41 .. synth_45): Advanced Segment Trees & Sweeps — SegTree Walk, GSS3 Max Subarray, Offline <=X Sum, Range Mod Beats, Contained Intervals."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_09() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_41: CANAL SLUICES (1800) - SegTree Binary Walk (First >= V in [L..N])
    # ------------------------------------------------------------------
    def gen_41(n: int, q: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, a))]
        for i in range(q):
            if i < q - 1 and r.random() < 0.35:
                lines.append(f"1 {r.randint(1, n)} {r.randint(1, vmax)}")
            else:
                l = r.randint(1, max(1, n // 3)) if r.random() < 0.7 else r.randint(1, n)
                v = r.randint(int(vmax * 0.85), vmax + max(1, vmax // 20))
                lines.append(f"2 {l} {v}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_41",
        title="Canal Sluices",
        rating=1800,
        tags=["data structures", "segment tree", "binary search"],
        notes="Max segment tree with point updates and O(log N) tree descent for smallest index >= L with H[i] >= V.",
        story_and_spec="""Along a canal stand N sluice gates numbered 1 to N, where gate i initially has capacity H_i. Process Q operations of two types:
- 1 P X: set the capacity of gate P to X.
- 2 L V: find the smallest gate index i such that i >= L and H_i >= V, or report -1 if no such gate exists.

Input
The first line contains N and Q.
The second line contains H_1 ... H_N.
Each of the next Q lines describes an operation in one of the two formats above. At least one operation is of type 2.

Output
For each operation of type 2, print the smallest valid index i (or -1) on its own line.

Constraints
1 <= N, Q <= 200000
1 <= H_i, X, V <= 10^9
1 <= P, L <= N""",
        samples=[
            "6 5\n3 7 2 9 5 8\n2 1 6\n2 3 6\n1 4 4\n2 3 6\n2 5 10\n",
        ],
        expected_samples=["2\n4\n6\n-1\n"],
        small_inputs=[
            "1 3\n5\n2 1 5\n1 1 3\n2 1 5\n",
            gen_41(400, 500, 100, 4101),
            gen_41(1500, 1600, 10**9, 4102),
        ],
        large_inputs=[
            # High thresholds force naive scan to walk almost to N every query
            (lambda: (
                lambda n, q, r: (
                    f"{n} {q}\n" + " ".join(str(r.randint(1, 1000)) for _ in range(n)) + "\n"
                    + "\n".join(
                        (f"1 {r.randint(1, n)} {r.randint(1, 1005)}" if k % 5 == 0
                         else f"2 {r.randint(1, 500)} {r.randint(998, 1010)}")
                        for k in range(q)
                    ) + "\n"
                )
            )(200000, 200000, random.Random(4111)))(),
            gen_41(200000, 200000, 10**9, 4112),
            gen_41(200000, 200000, 10000, 4113),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    vector<int> mx;
    SegTree(const vector<int>& a) {
        n = (int)a.size() - 1;
        mx.assign(4 * n + 4, 0);
        build(1, 1, n, a);
    }
    void build(int u, int l, int r, const vector<int>& a) {
        if (l == r) { mx[u] = a[l]; return; }
        int m = (l + r) >> 1;
        build(2 * u, l, m, a);
        build(2 * u + 1, m + 1, r, a);
        mx[u] = max(mx[2 * u], mx[2 * u + 1]);
    }
    void set_val(int u, int l, int r, int p, int v) {
        if (l == r) { mx[u] = v; return; }
        int m = (l + r) >> 1;
        if (p <= m) set_val(2 * u, l, m, p, v);
        else set_val(2 * u + 1, m + 1, r, p, v);
        mx[u] = max(mx[2 * u], mx[2 * u + 1]);
    }
    int find_first(int u, int l, int r, int ql, int v) const {
        if (r < ql || mx[u] < v) return -1;
        if (l == r) return l;
        int m = (l + r) >> 1;
        int res = find_first(2 * u, l, m, ql, v);
        if (res != -1) return res;
        return find_first(2 * u + 1, m + 1, r, ql, v);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    SegTree st(a);
    while (q--) {
        int t, x, y;
        cin >> t >> x >> y;
        if (t == 1) st.set_val(1, 1, n, x, y);
        else cout << st.find_first(1, 1, n, x, y) << '\n';
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int t, x, y;
        cin >> t >> x >> y;
        if (t == 1) a[x] = y;
        else {
            int ans = -1;
            for (int i = x; i <= n; i++) if (a[i] >= y) { ans = i; break; }
            cout << ans << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_42: BANNER ALIGNMENTS (1900) - Dynamic Range Max Non-Empty Subarray Sum
    # ------------------------------------------------------------------
    def gen_42(n: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(-10**9, 10**9) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, a))]
        for i in range(q):
            if i < q - 1 and r.random() < 0.35:
                lines.append(f"1 {r.randint(1, n)} {r.randint(-10**9, 10**9)}")
            else:
                L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
                if r.random() < 0.65:
                    L = r.randint(1, max(1, n // 5))
                    R = r.randint(max(L, 4 * n // 5), n)
                lines.append(f"2 {L} {R}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_42",
        title="Banner Alignments",
        rating=1900,
        tags=["data structures", "segment tree", "divide and conquer", "dp"],
        notes="Segment tree storing (total_sum, max_pref, max_suff, max_subarray) for non-empty subarrays with point updates.",
        story_and_spec="""N legion cohorts stand in formation numbered 1 to N, where cohort i initially has morale rating A_i (which may be negative). Process Q operations of two types:
- 1 P X: update the morale rating of cohort P to X.
- 2 L R: choose a NON-EMPTY contiguous subsegment [u, v] satisfying L <= u <= v <= R to maximize the sum A_u + A_{u+1} + ... + A_v.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines describes an operation in one of the two formats above. At least one operation is of type 2.

Output
For each operation of type 2, print the maximum non-empty subarray sum on its own line.

Constraints
1 <= N, Q <= 200000
-10^9 <= A_i, X <= 10^9
1 <= P <= N
1 <= L <= R <= N""",
        samples=[
            "5 4\n-2 4 -1 3 -5\n2 1 5\n1 3 -10\n2 1 5\n2 5 5\n",
        ],
        expected_samples=["6\n4\n-5\n"],
        small_inputs=[
            "3 2\n-8 -3 -5\n2 1 3\n2 1 1\n",
            gen_42(400, 500, 4201),
            gen_42(1500, 1500, 4202),
        ],
        large_inputs=[
            gen_42(180000, 180000, 4211),
            gen_42(200000, 200000, 4212),
            gen_42(200000, 200000, 4213),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long NEG = -(1LL << 60);
struct Node {
    long long sum, pref, suff, best;
};

static inline Node make_leaf(long long v) { return {v, v, v, v}; }
static inline Node merge_node(const Node& a, const Node& b) {
    return {
        a.sum + b.sum,
        max(a.pref, a.sum + b.pref),
        max(b.suff, b.sum + a.suff),
        max({a.best, b.best, a.suff + b.pref})
    };
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    int sz = 1;
    while (sz < n) sz <<= 1;
    vector<Node> seg(2 * sz, {0, NEG, NEG, NEG});
    for (int i = 0; i < n; i++) {
        long long x; cin >> x;
        seg[sz + i] = make_leaf(x);
    }
    for (int i = sz - 1; i >= 1; i--) seg[i] = merge_node(seg[2 * i], seg[2 * i + 1]);

    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int p; long long x;
            cin >> p >> x;
            int u = sz + p - 1;
            seg[u] = make_leaf(x);
            for (u >>= 1; u >= 1; u >>= 1) seg[u] = merge_node(seg[2 * u], seg[2 * u + 1]);
        } else {
            int l, r;
            cin >> l >> r;
            Node L{0, NEG, NEG, NEG}, R{0, NEG, NEG, NEG};
            bool hasL = false, hasR = false;
            for (l += sz - 1, r += sz; l < r; l >>= 1, r >>= 1) {
                if (l & 1) { L = hasL ? merge_node(L, seg[l]) : seg[l]; hasL = true; l++; }
                if (r & 1) { --r; R = hasR ? merge_node(seg[r], R) : seg[r]; hasR = true; }
            }
            Node res = !hasL ? R : (!hasR ? L : merge_node(L, R));
            cout << res.best << '\n';
        }
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int p; long long x; cin >> p >> x;
            a[p] = x;
        } else {
            int l, r; cin >> l >> r;
            long long cur = a[l], best = a[l];
            for (int i = l + 1; i <= r; i++) {
                cur = max(a[i], cur + a[i]);
                best = max(best, cur);
            }
            cout << best << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_43: ORCHARD HARVEST (1900) - Offline Subarray Sum of Elements <= X
    # ------------------------------------------------------------------
    def gen_43(n: int, q: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        qs = []
        for _ in range(q):
            L, R = r.randint(1, n), r.randint(1, n)
            if L > R:
                L, R = R, L
            qs.append(f"{L} {R} {r.randint(1, vmax)}")
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_43",
        title="Orchard Harvest",
        rating=1900,
        tags=["data structures", "fenwick tree", "offline", "sorting"],
        notes="Sort array elements and queries in non-decreasing order of value/threshold X; insert A_i into a Fenwick tree and query range sums.",
        story_and_spec="""N crates of apples sit in a row numbered 1 to N, where crate i weighs A_i kilograms. You are asked Q grading queries: each query specifies L, R, and X, and asks for the sum of weights of all crates i in the range L <= i <= R satisfying A_i <= X.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L, R, and X.

Output
For each query print the total qualifying weight on its own line.

Constraints
1 <= N, Q <= 200000
1 <= A_i, X <= 10^9
1 <= L <= R <= N""",
        samples=[
            "5 4\n8 3 10 2 5\n1 5 5\n1 3 8\n2 4 2\n3 5 1\n",
        ],
        expected_samples=["10\n11\n2\n0\n"],
        small_inputs=[
            "1 2\n7\n1 1 6\n1 1 7\n",
            gen_43(400, 500, 50, 4301),
            gen_43(1500, 1500, 10**9, 4302),
        ],
        large_inputs=[
            gen_43(180000, 180000, 10**9, 4311),
            gen_43(200000, 200000, 10**9, 4312),
            gen_43(200000, 200000, 5000, 4313),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<pair<int, int>> items(n);
    for (int i = 0; i < n; i++) { cin >> items[i].first; items[i].second = i + 1; }
    struct Qry { int l, r, x, id; };
    vector<Qry> qs(q);
    for (int i = 0; i < q; i++) { cin >> qs[i].l >> qs[i].r >> qs[i].x; qs[i].id = i; }

    sort(items.begin(), items.end());
    sort(qs.begin(), qs.end(), [](const Qry& a, const Qry& b) { return a.x < b.x; });

    vector<long long> bit(n + 1, 0), ans(q);
    auto add = [&](int idx, long long v) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += v;
    };
    auto sum = [&](int idx) {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    int p = 0;
    for (const auto& qr : qs) {
        while (p < n && items[p].first <= qr.x) {
            add(items[p].second, items[p].first);
            p++;
        }
        ans[qr.id] = sum(qr.r) - sum(qr.l - 1);
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int l, r, x; cin >> l >> r >> x;
        long long s = 0;
        for (int i = l; i <= r; i++) if (a[i] <= x) s += a[i];
        cout << s << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_44: SIGNAL MODULATORS (2100) - Segment Tree Beats (Range Mod + Point Set + Sum)
    # ------------------------------------------------------------------
    def gen_44(n: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, 10**9) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, a))]
        for i in range(q):
            t = 3 if i == q - 1 else r.choices([1, 2, 3], weights=[40, 25, 35])[0]
            if t == 1:
                L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
                if r.random() < 0.7:
                    L = r.randint(1, max(1, n // 5))
                    R = r.randint(max(L, 4 * n // 5), n)
                lines.append(f"1 {L} {R} {r.randint(2, 10**9)}")
            elif t == 2:
                lines.append(f"2 {r.randint(1, n)} {r.randint(1, 10**9)}")
            else:
                L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
                lines.append(f"3 {L} {R}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_44",
        title="Signal Modulators",
        rating=2100,
        tags=["data structures", "segment tree", "segment tree beats", "number theory"],
        notes="Segment tree storing (sum, max). Prune range-modulo recursion whenever subtree max < M; since x mod M < x/2 when x >= M, amortized cost is O((N + Q) log N log V).",
        story_and_spec="""An array of N signal amplifiers is numbered 1 to N, where amplifier i initially outputs power A_i. Process Q operations of three types:
- 1 L R M: for every i from L to R, replace A_i with (A_i mod M).
- 2 P X: set A_P to X.
- 3 L R: report the sum A_L + A_{L+1} + ... + A_R.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines describes an operation in one of the three formats above. At least one operation is of type 3.

Output
For each operation of type 3, print the range sum on its own line.

Constraints
1 <= N, Q <= 180000
0 <= A_i, X <= 10^9
1 <= M <= 10^9
1 <= P <= N
1 <= L <= R <= N""",
        samples=[
            "5 5\n10 7 15 4 9\n3 1 5\n1 2 4 6\n3 1 5\n2 3 20\n3 2 5\n",
        ],
        expected_samples=["45\n27\n34\n"],
        small_inputs=[
            "2 3\n100 200\n1 1 2 1\n3 1 2\n2 1 5\n",
            gen_44(400, 500, 4401),
            gen_44(1400, 1500, 4402),
        ],
        large_inputs=[
            # Large M (around 10^9 + 1) makes naive loop scan full [L..R] while Beats exits at root!
            (lambda: (
                lambda n, q, r: (
                    f"{n} {q}\n" + " ".join(str(r.randint(1, 5 * 10**8)) for _ in range(n)) + "\n"
                    + "\n".join(
                        (f"1 {r.randint(1, 500)} {r.randint(n - 500, n)} {r.randint(6 * 10**8, 10**9)}" if k % 3 == 0
                         else (f"2 {r.randint(1, n)} {r.randint(1, 10**9)}" if k % 3 == 1
                               else f"3 {r.randint(1, n // 2)} {r.randint(n // 2 + 1, n)}"))
                        for k in range(q)
                    ) + "\n"
                )
            )(180000, 180000, random.Random(4411)))(),
            gen_44(180000, 180000, 4412),
            gen_44(180000, 180000, 4413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct SegBeats {
    int n;
    vector<long long> sm, mx;
    SegBeats(const vector<long long>& a) {
        n = (int)a.size() - 1;
        sm.assign(4 * n + 4, 0);
        mx.assign(4 * n + 4, 0);
        build(1, 1, n, a);
    }
    void pull(int u) {
        sm[u] = sm[2 * u] + sm[2 * u + 1];
        mx[u] = max(mx[2 * u], mx[2 * u + 1]);
    }
    void build(int u, int l, int r, const vector<long long>& a) {
        if (l == r) { sm[u] = mx[u] = a[l]; return; }
        int m = (l + r) >> 1;
        build(2 * u, l, m, a);
        build(2 * u + 1, m + 1, r, a);
        pull(u);
    }
    void range_mod(int u, int l, int r, int ql, int qr, long long mod) {
        if (r < ql || qr < l || mx[u] < mod) return;
        if (l == r) { sm[u] %= mod; mx[u] = sm[u]; return; }
        int m = (l + r) >> 1;
        range_mod(2 * u, l, m, ql, qr, mod);
        range_mod(2 * u + 1, m + 1, r, ql, qr, mod);
        pull(u);
    }
    void point_set(int u, int l, int r, int p, long long v) {
        if (l == r) { sm[u] = mx[u] = v; return; }
        int m = (l + r) >> 1;
        if (p <= m) point_set(2 * u, l, m, p, v);
        else point_set(2 * u + 1, m + 1, r, p, v);
        pull(u);
    }
    long long query_sum(int u, int l, int r, int ql, int qr) const {
        if (r < ql || qr < l) return 0;
        if (ql <= l && r <= qr) return sm[u];
        int m = (l + r) >> 1;
        return query_sum(2 * u, l, m, ql, qr) + query_sum(2 * u + 1, m + 1, r, ql, qr);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    SegBeats st(a);
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int l, r; long long m;
            cin >> l >> r >> m;
            st.range_mod(1, 1, n, l, r, m);
        } else if (t == 2) {
            int p; long long x;
            cin >> p >> x;
            st.point_set(1, 1, n, p, x);
        } else {
            int l, r;
            cin >> l >> r;
            cout << st.query_sum(1, 1, n, l, r) << '\n';
        }
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int l, r; long long m; cin >> l >> r >> m;
            for (int i = l; i <= r; i++) a[i] %= m;
        } else if (t == 2) {
            int p; long long x; cin >> p >> x;
            a[p] = x;
        } else {
            int l, r; cin >> l >> r;
            long long s = 0;
            for (int i = l; i <= r; i++) s += a[i];
            cout << s << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_45: CARAVAN CONTRACTS (1900) - Offline Contained-Interval Sum
    # ------------------------------------------------------------------
    def gen_45(n: int, m: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {m} {q}"]
        for _ in range(m):
            l, rr = r.randint(1, n), r.randint(1, n)
            if l > rr:
                l, rr = rr, l
            lines.append(f"{l} {rr} {r.randint(1, 10**6)}")
        for _ in range(q):
            a, b = r.randint(1, n), r.randint(1, n)
            if a > b:
                a, b = b, a
            lines.append(f"{a} {b}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_45",
        title="Caravan Contracts",
        rating=1900,
        tags=["data structures", "fenwick tree", "offline", "sweep line"],
        notes="Sweep right endpoint B from 1..N; add contract weight W_i at index L_i when R_i == B, then sum [A..B] via Fenwick tree.",
        story_and_spec="""Over a trading season of N days (numbered 1 to N), a merchant guild has signed M caravan contracts. Contract i starts on day L_i, ends on day R_i, and yields profit W_i.

You are asked Q audit queries. Query j gives an interval [A, B]: compute the total profit of all contracts i that are COMPLETELY CONTAINED within [A, B] (that is, A <= L_i <= R_i <= B).

Input
The first line contains N, M, and Q.
Each of the next M lines contains L_i, R_i, and W_i.
Each of the next Q lines contains A and B.

Output
For each audit query print the total contained profit on its own line.

Constraints
1 <= N, M, Q <= 200000
1 <= L_i <= R_i <= N
1 <= W_i <= 10^6
1 <= A <= B <= N""",
        samples=[
            "6 5 3\n1 3 10\n2 4 20\n2 3 5\n4 6 15\n1 6 100\n1 4\n2 6\n1 6\n",
        ],
        expected_samples=["35\n40\n150\n"],
        small_inputs=[
            "3 2 2\n1 3 7\n2 2 4\n2 2\n1 3\n",
            gen_45(400, 500, 500, 4501),
            gen_45(1500, 1500, 1500, 4502),
        ],
        large_inputs=[
            gen_45(180000, 180000, 180000, 4511),
            gen_45(200000, 200000, 200000, 4512),
            gen_45(200000, 200000, 200000, 4513),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, long long>>> by_r(n + 1);
    for (int i = 0; i < m; i++) {
        int l, r; long long w;
        cin >> l >> r >> w;
        by_r[r].push_back({l, w});
    }
    vector<vector<pair<int, int>>> qs(n + 1);
    for (int i = 0; i < q; i++) {
        int a, b;
        cin >> a >> b;
        qs[b].push_back({a, i});
    }
    vector<long long> bit(n + 1, 0), ans(q);
    auto add = [&](int idx, long long v) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += v;
    };
    auto sum = [&](int idx) {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };
    long long total = 0;
    for (int r = 1; r <= n; r++) {
        for (auto [l, w] : by_r[r]) { add(l, w); total += w; }
        for (auto [a, id] : qs[r]) ans[id] = total - sum(a - 1);
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<int> l(m), r(m);
    vector<long long> w(m);
    for (int i = 0; i < m; i++) cin >> l[i] >> r[i] >> w[i];
    while (q--) {
        int a, b; cin >> a >> b;
        long long s = 0;
        for (int i = 0; i < m; i++) if (a <= l[i] && r[i] <= b) s += w[i];
        cout << s << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_09())
