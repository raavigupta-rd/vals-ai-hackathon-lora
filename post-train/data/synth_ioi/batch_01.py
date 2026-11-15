"""Batch 1 (synth_01 .. synth_05): Offline Sweeps & Segment/Fenwick Trees."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_01() -> list[ProblemSpec]:
    rng = random.Random(101)
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_01: ORCHARD MEX (1800) - Offline R-sweep + Min-Pos SegTree
    # ------------------------------------------------------------------
    def gen_01(n: int, q: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(0, vmax) for _ in range(n)]
        qs = []
        for _ in range(q):
            x, y = r.randint(1, n), r.randint(1, n)
            if x > y:
                x, y = y, x
            qs.append(f"{x} {y}")
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_01",
        title="Orchard MEX",
        rating=1800,
        tags=["data structures", "segment tree", "offline"],
        notes="Offline sweep by R; segment tree over values [0..N] storing last seen index, binary walk for first value with last_pos < L.",
        story_and_spec="""An orchard row has N fruit trees numbered 1 to N. Tree i bears fruit of variety A_i (a non-negative integer). You are asked Q questions. Each question gives two integers L and R, and asks for the smallest non-negative integer variety (0, 1, 2, ...) that does NOT appear anywhere among trees L, L+1, ..., R.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L and R.

Output
For each question print the smallest missing non-negative integer on its own line.

Constraints
1 <= N, Q <= 200000
0 <= A_i <= N
1 <= L <= R <= N""",
        samples=[
            "6 4\n0 2 1 0 3 1\n1 3\n2 5\n1 6\n5 6\n",
        ],
        expected_samples=["3\n4\n4\n0\n"],
        small_inputs=[
            "1 2\n0\n1 1\n1 1\n",
            "5 5\n1 2 3 4 5\n1 5\n1 1\n3 4\n2 5\n5 5\n",
            gen_01(400, 500, 15, 11),
            gen_01(1200, 1500, 400, 12),
            gen_01(1800, 1800, 1800, 13),
        ],
        large_inputs=[
            gen_01(180000, 180000, 50000, 21),
            gen_01(200000, 200000, 200000, 22),
            # Adversarial: permutation of 0..N-1 with wide queries
            (lambda: (
                lambda n, q, r: (
                    f"{n} {q}\n"
                    + " ".join(map(str, r.sample(range(n), n))) + "\n"
                    + "\n".join(f"{r.randint(1, n // 4)} {r.randint(3 * n // 4, n)}" for _ in range(q)) + "\n"
                )
            )(190000, 190000, random.Random(23)))(),
            gen_01(200000, 200000, 500, 24),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int sz;
    vector<int> mn;
    SegTree(int n) {
        sz = 1;
        while (sz <= n + 1) sz <<= 1;
        mn.assign(2 * sz, 0);
    }
    void update(int idx, int val) {
        int u = idx + sz;
        mn[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) mn[u] = min(mn[2 * u], mn[2 * u + 1]);
    }
    int query_first_lt(int thresh) const {
        int u = 1;
        while (u < sz) {
            if (mn[2 * u] < thresh) u = 2 * u;
            else u = 2 * u + 1;
        }
        return u - sz;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<pair<int, int>>> by_r(n + 1);
    for (int i = 0; i < q; i++) {
        int l, r;
        cin >> l >> r;
        by_r[r].push_back({l, i});
    }
    SegTree st(n + 1);
    vector<int> ans(q);
    for (int r = 1; r <= n; r++) {
        if (a[r] <= n + 1) st.update(a[r], r);
        for (auto [l, id] : by_r[r]) {
            ans[id] = st.query_first_lt(l);
        }
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
    vector<int> seen(n + 3, 0);
    for (int t = 1; t <= q; t++) {
        int l, r;
        cin >> l >> r;
        for (int i = l; i <= r; i++) if (a[i] <= n + 1) seen[a[i]] = t;
        int mex = 0;
        while (seen[mex] == t) mex++;
        cout << mex << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_02: DOMINANT BANNER (1900) - Dynamic Majority w/ Point Updates
    # ------------------------------------------------------------------
    def gen_02(n: int, q: int, cmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, cmax) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, a))]
        for _ in range(q):
            if r.random() < 0.35:
                p = r.randint(1, n)
                x = r.randint(1, cmax)
                lines.append(f"1 {p} {x}")
            else:
                L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
                lines.append(f"2 {L} {R}")
        # Ensure at least one type-2 query
        lines[-1] = f"2 1 {n}"
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_02",
        title="Dominant Banner",
        rating=1900,
        tags=["data structures", "segment tree", "divide and conquer"],
        notes="Boyer-Moore majority candidate segment tree + per-color coordinate-compressed Fenwick tree for exact count verification.",
        story_and_spec="""N guards stand in a line numbered 1 to N. Initially, guard i holds clan banner A_i. You process Q events of two types:
- 1 P X: guard P replaces their banner with banner X.
- 2 L R: among guards L, L+1, ..., R, find the clan banner held by STRICTLY MORE than (R - L + 1) / 2 guards, or report 0 if no such banner exists.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines describes an event in one of the two formats above. At least one event is of type 2.

Output
For each event of type 2, print the strictly dominant banner number, or 0 if none exists, on its own line.

Constraints
1 <= N, Q <= 200000
1 <= A_i, X <= N
1 <= P <= N
1 <= L <= R <= N""",
        samples=[
            "6 5\n2 1 2 2 3 1\n2 1 4\n2 2 6\n1 5 2\n2 2 6\n2 5 6\n",
        ],
        expected_samples=["2\n0\n2\n0\n"],
        small_inputs=[
            "1 3\n1\n2 1 1\n1 1 1\n2 1 1\n",
            gen_02(300, 400, 3, 201),
            gen_02(1000, 1200, 5, 202),
            gen_02(1600, 1600, 2, 203),
            gen_02(1500, 1500, 50, 204),
        ],
        large_inputs=[
            gen_02(180000, 180000, 2, 211),
            gen_02(200000, 200000, 4, 212),
            gen_02(200000, 200000, 500, 213),
            # Adversarial heavy updates + long queries
            (lambda: (
                lambda n, q, r: (
                    f"{n} {q}\n"
                    + " ".join(str(1 if i % 2 == 0 else r.randint(2, 5)) for i in range(n)) + "\n"
                    + "\n".join(
                        (f"1 {r.randint(1, n)} {r.choice([1, 2, 3])}" if k % 3 == 0
                         else f"2 {r.randint(1, n // 5)} {r.randint(4 * n // 5, n)}")
                        for k in range(q)
                    ) + "\n"
                )
            )(190000, 190000, random.Random(214)))(),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct Node {
    int cand = 0, cnt = 0;
};

static inline Node merge_node(const Node& a, const Node& b) {
    if (a.cand == b.cand) return {a.cand, a.cnt + b.cnt};
    if (a.cnt >= b.cnt) return {a.cand, a.cnt - b.cnt};
    return {b.cand, b.cnt - a.cnt};
}

struct BIT {
    vector<int> t;
    void init(int n) { t.assign(n + 1, 0); }
    void add(int i, int delta) {
        for (++i; i < (int)t.size(); i += i & -i) t[i] += delta;
    }
    int sum_prefix(int i) const { // sum over [0..i-1]
        int s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> init_a(n + 1);
    vector<vector<int>> positions(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> init_a[i];
        positions[init_a[i]].push_back(i);
    }
    struct Op { int type, a, b; };
    vector<Op> ops(q);
    for (int i = 0; i < q; i++) {
        cin >> ops[i].type >> ops[i].a >> ops[i].b;
        if (ops[i].type == 1) positions[ops[i].b].push_back(ops[i].a);
    }
    vector<BIT> bits(n + 1);
    for (int c = 1; c <= n; c++) {
        auto& v = positions[c];
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
        bits[c].init((int)v.size());
    }
    auto add_color = [&](int c, int pos, int delta) {
        const auto& v = positions[c];
        int idx = (int)(lower_bound(v.begin(), v.end(), pos) - v.begin());
        bits[c].add(idx, delta);
    };
    auto count_range = [&](int c, int l, int r) -> int {
        if (c <= 0 || c > n) return 0;
        const auto& v = positions[c];
        int ri = (int)(upper_bound(v.begin(), v.end(), r) - v.begin());
        int li = (int)(lower_bound(v.begin(), v.end(), l) - v.begin());
        return bits[c].sum_prefix(ri) - bits[c].sum_prefix(li);
    };

    int sz = 1;
    while (sz < n) sz <<= 1;
    vector<Node> seg(2 * sz);
    vector<int> cur = init_a;
    for (int i = 1; i <= n; i++) {
        seg[sz + i - 1] = {cur[i], 1};
        add_color(cur[i], i, +1);
    }
    for (int i = sz - 1; i >= 1; i--) seg[i] = merge_node(seg[2 * i], seg[2 * i + 1]);

    auto seg_set = [&](int pos, int val) {
        int u = sz + pos - 1;
        seg[u] = {val, 1};
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = merge_node(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_query = [&](int l, int r) {
        Node L{}, R{};
        for (l += sz - 1, r += sz; l < r; l >>= 1, r >>= 1) {
            if (l & 1) L = merge_node(L, seg[l++]);
            if (r & 1) R = merge_node(seg[--r], R);
        }
        return merge_node(L, R).cand;
    };

    for (const auto& op : ops) {
        if (op.type == 1) {
            int p = op.a, x = op.b;
            if (cur[p] != x) {
                add_color(cur[p], p, -1);
                cur[p] = x;
                add_color(cur[p], p, +1);
                seg_set(p, x);
            }
        } else {
            int l = op.a, r = op.b;
            int cand = seg_query(l, r);
            int freq = count_range(cand, l, r);
            cout << (freq * 2 > (r - l + 1) ? cand : 0) << '\n';
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
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<int> cnt(n + 1, 0), touched;
    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int p, x;
            cin >> p >> x;
            a[p] = x;
        } else {
            int l, r;
            cin >> l >> r;
            int ans = 0, need = (r - l + 1) / 2;
            touched.clear();
            for (int i = l; i <= r; i++) {
                if (cnt[a[i]] == 0) touched.push_back(a[i]);
                if (++cnt[a[i]] > need) ans = a[i];
            }
            for (int c : touched) cnt[c] = 0;
            cout << ans << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_03: FLOODED TERRACES (2000) - Subarray Rainwater Queries
    # ------------------------------------------------------------------
    def gen_03(n: int, q: int, hmax: int, seed: int) -> str:
        r = random.Random(seed)
        h = [r.randint(1, hmax) for _ in range(n)]
        qs = []
        for _ in range(q):
            L, R = r.randint(1, n), r.randint(1, n)
            if L > R:
                L, R = R, L
            qs.append(f"{L} {R}")
        return f"{n} {q}\n" + " ".join(map(str, h)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_03",
        title="Flooded Terraces",
        rating=2000,
        tags=["data structures", "monotonic stack", "binary lifting", "rmq"],
        notes="Find leftmost and rightmost maximum in [L, R] via Sparse Table RMQ, then binary-lift along next-greater-right from L and prev-greater-left from R.",
        story_and_spec="""Along a hillside stand N stone walls numbered 1 to N, where wall i has height H_i. You are asked Q independent questions. Each question specifies L and R: imagine that only walls L, L+1, ..., R exist (water spills freely off to the left of L and to the right of R). After heavy rain, water settles above each wall i (L <= i <= R) up to the height min(max_{L <= j <= i} H_j, max_{i <= j <= R} H_j). Find the total volume of trapped water summed over all i from L to R.

Input
The first line contains N and Q.
The second line contains H_1 ... H_N.
Each of the next Q lines contains L and R.

Output
For each question print the total trapped water volume on its own line.

Constraints
1 <= N, Q <= 200000
1 <= H_i <= 10^9
1 <= L <= R <= N""",
        samples=[
            "7 4\n4 1 3 2 5 1 3\n1 5\n2 7\n3 6\n4 4\n",
        ],
        expected_samples=["6\n3\n1\n0\n"],
        small_inputs=[
            "5 4\n3 1 3 2 3\n1 5\n1 3\n2 4\n3 5\n",
            gen_03(300, 400, 10, 301),
            gen_03(1200, 1500, 1000000000, 302),
            gen_03(1700, 1700, 5, 303),
            # Strictly increasing + V-shape
            "6 3\n10 5 1 1 5 10\n1 6\n2 5\n1 4\n",
        ],
        large_inputs=[
            gen_03(180000, 180000, 1000000000, 311),
            gen_03(200000, 200000, 50, 312),
            # Adversarial sawtooth / staircase forcing O(N) jumps without binary lifting
            (lambda: (
                lambda n, q, r: (
                    f"{n} {q}\n"
                    + " ".join(str((i * 7 + (i % 3) * 2) + 1) for i in range(n)) + "\n"
                    + "\n".join(f"{r.randint(1, n // 6)} {r.randint(5 * n // 6, n)}" for _ in range(q)) + "\n"
                )
            )(190000, 190000, random.Random(313)))(),
            gen_03(200000, 200000, 1000000, 314),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> h(n + 1), pref(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        cin >> h[i];
        pref[i] = pref[i - 1] + h[i];
    }

    int LOG = 1;
    while ((1 << LOG) <= n + 1) LOG++;

    // Sparse tables for leftmost max and rightmost max
    vector<vector<int>> stL(LOG, vector<int>(n + 1)), stR(LOG, vector<int>(n + 1));
    for (int i = 1; i <= n; i++) stL[0][i] = stR[0][i] = i;
    for (int k = 1; k < LOG; k++) {
        int len = 1 << (k - 1);
        for (int i = 1; i + (1 << k) - 1 <= n; i++) {
            int a = stL[k - 1][i], b = stL[k - 1][i + len];
            stL[k][i] = (h[a] >= h[b] ? a : b);
            int c = stR[k - 1][i], d = stR[k - 1][i + len];
            stR[k][i] = (h[d] >= h[c] ? d : c);
        }
    }
    auto query_bounds = [&](int l, int r) -> pair<int, int> {
        int k = __lg(r - l + 1);
        int a = stL[k][l], b = stL[k][r - (1 << k) + 1];
        int ml = (h[a] >= h[b] ? a : b);
        int c = stR[k][l], d = stR[k][r - (1 << k) + 1];
        int mr = (h[d] >= h[c] ? d : c);
        return {ml, mr};
    };

    // Next strictly greater to the right (1..n, sentinel n+1)
    vector<int> nxt(n + 2, n + 1);
    {
        vector<int> st;
        for (int i = n; i >= 1; i--) {
            while (!st.empty() && h[st.back()] <= h[i]) st.pop_back();
            nxt[i] = st.empty() ? n + 1 : st.back();
            st.push_back(i);
        }
    }
    // Previous strictly greater to the left (1..n, sentinel 0)
    vector<int> prv(n + 2, 0);
    {
        vector<int> st;
        for (int i = 1; i <= n; i++) {
            while (!st.empty() && h[st.back()] <= h[i]) st.pop_back();
            prv[i] = st.empty() ? 0 : st.back();
            st.push_back(i);
        }
    }

    vector<vector<int>> upR(LOG, vector<int>(n + 2, n + 1));
    vector<vector<long long>> sumR(LOG, vector<long long>(n + 2, 0));
    for (int i = 1; i <= n; i++) {
        upR[0][i] = nxt[i];
        if (nxt[i] <= n) sumR[0][i] = 1LL * (nxt[i] - i) * h[i];
    }
    for (int k = 1; k < LOG; k++) {
        for (int i = 1; i <= n; i++) {
            int mid = upR[k - 1][i];
            upR[k][i] = upR[k - 1][mid];
            sumR[k][i] = sumR[k - 1][i] + sumR[k - 1][mid];
        }
    }

    vector<vector<int>> upL(LOG, vector<int>(n + 2, 0));
    vector<vector<long long>> sumL(LOG, vector<long long>(n + 2, 0));
    for (int i = 1; i <= n; i++) {
        upL[0][i] = prv[i];
        if (prv[i] >= 1) sumL[0][i] = 1LL * (i - prv[i]) * h[i];
    }
    for (int k = 1; k < LOG; k++) {
        for (int i = 1; i <= n; i++) {
            int mid = upL[k - 1][i];
            upL[k][i] = upL[k - 1][mid];
            sumL[k][i] = sumL[k - 1][i] + sumL[k - 1][mid];
        }
    }

    while (q--) {
        int l, r;
        cin >> l >> r;
        auto [ml, mr] = query_bounds(l, r);
        long long total_ceil = 0;
        int u = l;
        for (int k = LOG - 1; k >= 0; k--) {
            if (upR[k][u] <= ml) {
                total_ceil += sumR[k][u];
                u = upR[k][u];
            }
        }
        int v = r;
        for (int k = LOG - 1; k >= 0; k--) {
            if (upL[k][v] >= mr) {
                total_ceil += sumL[k][v];
                v = upL[k][v];
            }
        }
        total_ceil += 1LL * (mr - ml + 1) * h[ml];
        cout << (total_ceil - (pref[r] - pref[l - 1])) << '\n';
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
    vector<long long> h(n + 1);
    for (int i = 1; i <= n; i++) cin >> h[i];
    vector<long long> pref(n + 2), suf(n + 2);
    while (q--) {
        int l, r;
        cin >> l >> r;
        long long mx = 0;
        for (int i = l; i <= r; i++) {
            mx = max(mx, h[i]);
            pref[i] = mx;
        }
        mx = 0;
        long long ans = 0;
        for (int i = r; i >= l; i--) {
            mx = max(mx, h[i]);
            ans += min(pref[i], mx) - h[i];
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_04: SIGNAL RELAYS (1900) - 2D Rectangle Sum Sweepline
    # ------------------------------------------------------------------
    def gen_04(n: int, q: int, cmax: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {q}"]
        for _ in range(n):
            lines.append(f"{r.randint(1, cmax)} {r.randint(1, cmax)} {r.randint(1, 10**9)}")
        for _ in range(q):
            x1, x2 = r.randint(1, cmax), r.randint(1, cmax)
            y1, y2 = r.randint(1, cmax), r.randint(1, cmax)
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            lines.append(f"{x1} {y1} {x2} {y2}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_04",
        title="Signal Relays",
        rating=1900,
        tags=["data structures", "sweep line", "fenwick tree", "coordinate compression"],
        notes="4-corner prefix split on X, sweep sorted events with coordinate-compressed Fenwick tree on Y.",
        story_and_spec="""Across a desert plain are N signal relay towers. Tower i stands at integer coordinates (X_i, Y_i) and has transmission capacity W_i (multiple towers may share the same coordinates). You are asked Q sector queries. Each query specifies X_1, Y_1, X_2, Y_2 and asks for the sum of capacities W_i over all towers satisfying X_1 <= X_i <= X_2 and Y_1 <= Y_i <= Y_2.

Input
The first line contains N and Q.
Each of the next N lines contains X_i, Y_i, and W_i.
Each of the next Q lines contains X_1, Y_1, X_2, and Y_2.

Output
For each sector query print the total capacity on its own line.

Constraints
1 <= N, Q <= 200000
1 <= X_i, Y_i, W_i <= 10^9
1 <= X_1 <= X_2 <= 10^9
1 <= Y_1 <= Y_2 <= 10^9""",
        samples=[
            "5 3\n1 2 10\n3 4 20\n3 2 5\n6 5 100\n1 2 7\n1 1 3 4\n2 2 6 4\n5 5 5 5\n",
        ],
        expected_samples=["42\n25\n0\n"],
        small_inputs=[
            "1 2\n5 5 1000000000\n5 5 5 5\n1 1 4 10\n",
            gen_04(400, 500, 20, 401),
            gen_04(1400, 1500, 10**9, 402),
            gen_04(1600, 1600, 500, 403),
        ],
        large_inputs=[
            gen_04(180000, 180000, 10**9, 411),
            gen_04(200000, 200000, 2000, 412),
            gen_04(200000, 200000, 10**9, 413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    struct Pt { int x, y; long long w; };
    vector<Pt> pts(n);
    vector<int> ys;
    ys.reserve(n);
    for (int i = 0; i < n; i++) {
        cin >> pts[i].x >> pts[i].y >> pts[i].w;
        ys.push_back(pts[i].y);
    }
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());
    int m = (int)ys.size();

    struct Ev { int x, y1, y2, id, sign; };
    vector<Ev> evs;
    evs.reserve(2 * q);
    for (int i = 0; i < q; i++) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        int ly = (int)(lower_bound(ys.begin(), ys.end(), y1) - ys.begin());
        int ry = (int)(upper_bound(ys.begin(), ys.end(), y2) - ys.begin());
        if (ly < ry) {
            evs.push_back({x2, ly, ry, i, +1});
            evs.push_back({x1 - 1, ly, ry, i, -1});
        }
    }
    sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) { return a.x < b.x; });
    sort(evs.begin(), evs.end(), [](const Ev& a, const Ev& b) { return a.x < b.x; });

    vector<long long> bit(m + 1, 0);
    auto add = [&](int idx, long long v) {
        for (++idx; idx <= m; idx += idx & -idx) bit[idx] += v;
    };
    auto sum = [&](int idx) {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    vector<long long> ans(q, 0);
    int p = 0;
    for (const auto& e : evs) {
        while (p < n && pts[p].x <= e.x) {
            int yi = (int)(lower_bound(ys.begin(), ys.end(), pts[p].y) - ys.begin());
            add(yi, pts[p].w);
            p++;
        }
        ans[e.id] += e.sign * (sum(e.y2) - sum(e.y1));
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
    vector<int> x(n), y(n);
    vector<long long> w(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i] >> w[i];
    while (q--) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        long long s = 0;
        for (int i = 0; i < n; i++) {
            if (x1 <= x[i] && x[i] <= x2 && y1 <= y[i] && y[i] <= y2) s += w[i];
        }
        cout << s << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_05: TOLL REBELLION (2000) - Weighted Segment Union (Lazy Min+Sum)
    # ------------------------------------------------------------------
    def gen_05(n: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        w = [r.randint(1, 10**6) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, w))]
        active: list[tuple[int, int]] = []
        for _ in range(q):
            if not active or r.random() < 0.62:
                L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
                active.append((L, R))
                lines.append(f"1 {L} {R}")
            else:
                idx = r.randrange(len(active))
                L, R = active[idx]
                active[idx] = active[-1]
                active.pop()
                lines.append(f"2 {L} {R}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_05",
        title="Toll Rebellion",
        rating=2000,
        tags=["data structures", "lazy segment tree"],
        notes="Lazy range-add segment tree maintaining (minimum coverage count, sum of lengths achieving that minimum). Covered length = total_len - (min_val == 0 ? min_sum : 0).",
        story_and_spec="""A royal highway consists of N consecutive road segments numbered 1 to N, where segment i has length W_i kilometres. Initially there are no blockades on the highway. You process Q decrees in chronological order:
- 1 L R: a rebel guild erects a blockade covering segments L, L+1, ..., R.
- 2 L R: a previously erected blockade covering segments L, L+1, ..., R is lifted (it is guaranteed that at least one active blockade currently covers every segment from L to R; note that multiple blockades may overlap).

After each decree, print the total length in kilometres of segments that are currently covered by AT LEAST ONE active blockade.

Input
The first line contains N and Q.
The second line contains W_1 ... W_N.
Each of the next Q lines contains type, L, and R (type is 1 or 2).

Output
Print Q lines: after each decree, the total covered length in kilometres.

Constraints
1 <= N, Q <= 200000
1 <= W_i <= 10^6
1 <= L <= R <= N""",
        samples=[
            "4 5\n10 20 30 40\n1 1 2\n1 2 3\n2 1 2\n1 4 4\n2 2 3\n",
        ],
        expected_samples=["30\n60\n50\n90\n40\n"],
        small_inputs=[
            "1 4\n100\n1 1 1\n1 1 1\n2 1 1\n2 1 1\n",
            gen_05(300, 500, 501),
            gen_05(1400, 1600, 502),
            gen_05(1800, 1800, 503),
        ],
        large_inputs=[
            gen_05(180000, 180000, 511),
            gen_05(200000, 200000, 512),
            gen_05(200000, 200000, 513),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    vector<int> mn, lazy;
    vector<long long> msum;

    SegTree(const vector<long long>& w) {
        n = (int)w.size() - 1;
        mn.assign(4 * n + 4, 0);
        lazy.assign(4 * n + 4, 0);
        msum.assign(4 * n + 4, 0);
        build(1, 1, n, w);
    }
    void pull(int u) {
        int a = mn[2 * u], b = mn[2 * u + 1];
        mn[u] = min(a, b);
        msum[u] = (a == mn[u] ? msum[2 * u] : 0) + (b == mn[u] ? msum[2 * u + 1] : 0);
    }
    void build(int u, int l, int r, const vector<long long>& w) {
        if (l == r) {
            mn[u] = 0;
            msum[u] = w[l];
            return;
        }
        int m = (l + r) >> 1;
        build(2 * u, l, m, w);
        build(2 * u + 1, m + 1, r, w);
        pull(u);
    }
    void apply(int u, int d) {
        mn[u] += d;
        lazy[u] += d;
    }
    void push(int u) {
        if (lazy[u] != 0) {
            apply(2 * u, lazy[u]);
            apply(2 * u + 1, lazy[u]);
            lazy[u] = 0;
        }
    }
    void add(int u, int l, int r, int ql, int qr, int d) {
        if (ql <= l && r <= qr) {
            apply(u, d);
            return;
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) add(2 * u, l, m, ql, qr, d);
        if (qr > m) add(2 * u + 1, m + 1, r, ql, qr, d);
        pull(u);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> w(n + 1);
    long long total = 0;
    for (int i = 1; i <= n; i++) {
        cin >> w[i];
        total += w[i];
    }
    SegTree st(w);
    while (q--) {
        int t, l, r;
        cin >> t >> l >> r;
        st.add(1, 1, n, l, r, t == 1 ? +1 : -1);
        long long zero_len = (st.mn[1] == 0 ? st.msum[1] : 0LL);
        cout << (total - zero_len) << '\n';
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
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i];
    vector<int> cnt(n + 1, 0);
    while (q--) {
        int t, l, r;
        cin >> t >> l >> r;
        int d = (t == 1 ? 1 : -1);
        for (int i = l; i <= r; i++) cnt[i] += d;
        long long ans = 0;
        for (int i = 1; i <= n; i++) if (cnt[i] > 0) ans += w[i];
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_01())
