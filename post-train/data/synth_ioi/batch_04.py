"""Batch 4 (synth_16 .. synth_20): DP Optimizations — Li Chao CHT, Divide & Conquer DP, SOS DP, SegTree Interval Cover DP, Prefix-Suffix Knapsack."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_04() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_16: RIVER COURIERS (2100) - Li Chao Tree DP
    # ------------------------------------------------------------------
    def gen_16(n: int, xmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, xmax) for _ in range(n)]
        d = [r.randint(1, xmax) for _ in range(n)]
        f = [r.randint(0, 10**11) for _ in range(n)]
        return (f"{n}\n" + " ".join(map(str, a)) + "\n"
                + " ".join(map(str, d)) + "\n" + " ".join(map(str, f)) + "\n")

    specs.append(ProblemSpec(
        pid="synth_16",
        title="River Couriers",
        rating=2100,
        tags=["dp", "convex hull trick", "li chao tree", "data structures"],
        notes="dp[i] = F_i + min_{1 <= j < i} (dp[j] + A_j * D_i); non-monotone slopes & queries solved via Li Chao tree on [1, 10^6].",
        story_and_spec="""Along a river are N docks numbered 1 to N from upstream to downstream. A message starts at dock 1 (with initial cost 0) and must reach dock N by a sequence of boat hops that always move strictly downstream (from some dock j to a dock i > j).

Each dock i has three parameters: fuel rate A_i, current factor D_i, and harbour fee F_i. Sending a boat directly from dock j to dock i (j < i) costs A_j * D_i + F_i.

For every dock i = 2, 3, ..., N, find the minimum total cost for the message to reach dock i starting from dock 1.

Input
The first line contains N.
The second line contains A_1 ... A_N.
The third line contains D_1 ... D_N.
The fourth line contains F_1 ... F_N.

Output
Print N - 1 lines: on the k-th line print the minimum total cost to reach dock k + 1.

Constraints
2 <= N <= 200000
1 <= A_i, D_i <= 10^6
0 <= F_i <= 10^11""",
        samples=[
            "4\n5 2 8 1\n3 6 4 10\n10 5 7 2\n",
        ],
        small_inputs=[
            "2\n7 3\n4 9\n100 50\n",
            gen_16(400, 1000, 1601),
            gen_16(1800, 10**6, 1602),
            gen_16(2200, 5000, 1603),
        ],
        large_inputs=[
            gen_16(180000, 10**6, 1611),
            gen_16(200000, 10**6, 1612),
            gen_16(200000, 50000, 1613),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 62);
struct Line {
    long long m, b;
    long long eval(long long x) const { return m * x + b; }
};

struct LiChao {
    int XMAX;
    vector<Line> tr;
    LiChao(int xmax) : XMAX(xmax), tr(4 * xmax + 8, {0, INF}) {}
    void add_line(int u, int l, int r, Line nw) {
        int mid = (l + r) >> 1;
        bool lef = nw.eval(l) < tr[u].eval(l);
        bool md = nw.eval(mid) < tr[u].eval(mid);
        if (md) swap(tr[u], nw);
        if (l == r) return;
        if (lef != md) add_line(2 * u, l, mid, nw);
        else add_line(2 * u + 1, mid + 1, r, nw);
    }
    long long query(int u, int l, int r, int x) const {
        long long res = tr[u].eval(x);
        if (l == r) return res;
        int mid = (l + r) >> 1;
        if (x <= mid) return min(res, query(2 * u, l, mid, x));
        return min(res, query(2 * u + 1, mid + 1, r, x));
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n + 1), d(n + 1), f(n + 1), dp(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> a[i];
    for (int i = 1; i <= n; i++) cin >> d[i];
    for (int i = 1; i <= n; i++) cin >> f[i];

    LiChao lc(1000000);
    lc.add_line(1, 1, 1000000, {a[1], 0});
    for (int i = 2; i <= n; i++) {
        dp[i] = f[i] + lc.query(1, 1, 1000000, (int)d[i]);
        lc.add_line(1, 1, 1000000, {a[i], dp[i]});
        cout << dp[i] << '\n';
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
    int n;
    cin >> n;
    vector<long long> a(n + 1), d(n + 1), f(n + 1), dp(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> a[i];
    for (int i = 1; i <= n; i++) cin >> d[i];
    for (int i = 1; i <= n; i++) cin >> f[i];
    for (int i = 2; i <= n; i++) {
        long long best = (1LL << 62);
        for (int j = 1; j < i; j++) best = min(best, dp[j] + a[j] * d[i]);
        dp[i] = best + f[i];
        cout << dp[i] << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_17: CANAL LOCKS (2200) - Divide & Conquer DP Optimization
    # ------------------------------------------------------------------
    def gen_17(n: int, k: int, seed: int) -> str:
        r = random.Random(seed)
        w = [r.randint(1, 10000) for _ in range(n)]
        return f"{n} {k}\n" + " ".join(map(str, w)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_17",
        title="Canal Locks",
        rating=2200,
        tags=["dp", "divide and conquer", "convex hull trick"],
        notes="Partition N positive weights into K contiguous groups minimizing sum of (group_sum)^2; quadrangle inequality / monotone opt via D&C DP.",
        story_and_spec="""N freight barges arrive in a single file numbered 1 to N, where barge i has displacement W_i. You must divide the line of barges into exactly K non-empty contiguous batches to pass through a canal lock one batch at a time.

If a batch consists of barges L, L+1, ..., R, the water turbulence cost of locking that batch is (W_L + W_{L+1} + ... + W_R)^2. Find the minimum total turbulence cost summed across all K batches.

Input
The first line contains N and K.
The second line contains W_1 ... W_N.

Output
Print a single integer: the minimum possible total turbulence cost.

Constraints
1 <= K <= 20
K <= N <= 100000
1 <= W_i <= 10000""",
        samples=[
            "5 2\n3 1 4 1 5\n",
        ],
        small_inputs=[
            "3 3\n5 6 7\n",
            "6 1\n1 2 3 4 5 6\n",
            gen_17(500, 10, 1701),
            gen_17(1500, 15, 1702),
        ],
        large_inputs=[
            gen_17(90000, 18, 1711),
            gen_17(100000, 20, 1712),
            gen_17(100000, 16, 1713),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 62);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> p(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        p[i] = p[i - 1] + x;
    }
    vector<long long> prv(n + 1, INF), cur(n + 1, INF);
    for (int i = 1; i <= n; i++) prv[i] = p[i] * p[i];

    function<void(int, int, int, int)> dnc = [&](int l, int r, int optl, int optr) {
        if (l > r) return;
        int mid = (l + r) >> 1;
        pair<long long, int> best = {INF, optl};
        int hi = min(mid - 1, optr);
        for (int j = optl; j <= hi; j++) {
            long long d = p[mid] - p[j];
            long long val = prv[j] + d * d;
            if (val < best.first) best = {val, j};
        }
        cur[mid] = best.first;
        dnc(l, mid - 1, optl, best.second);
        dnc(mid + 1, r, best.second, optr);
    };

    for (int step = 2; step <= k; step++) {
        fill(cur.begin(), cur.end(), INF);
        dnc(step, n, step - 1, n - 1);
        prv.swap(cur);
    }
    cout << prv[n] << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> p(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        p[i] = p[i - 1] + x;
    }
    const long long INF = (1LL << 62);
    vector<long long> dp(n + 1, INF), ndp(n + 1, INF);
    dp[0] = 0;
    for (int step = 1; step <= k; step++) {
        fill(ndp.begin(), ndp.end(), INF);
        for (int i = step; i <= n; i++) {
            for (int j = step - 1; j < i; j++) {
                long long d = p[i] - p[j];
                ndp[i] = min(ndp[i], dp[j] + d * d);
            }
        }
        dp.swap(ndp);
    }
    cout << dp[n] << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_18: RUNE COMPATIBILITY (1900) - 20-bit SOS DP
    # ------------------------------------------------------------------
    def gen_18(n: int, q: int, bits: int, seed: int) -> str:
        r = random.Random(seed)
        lim = (1 << bits) - 1
        lines = [f"{n} {q}"]
        for _ in range(n):
            lines.append(f"{r.randint(0, lim)} {r.randint(1, 10**6)}")
        for _ in range(q):
            lines.append(str(r.randint(0, lim)))
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_18",
        title="Rune Compatibility",
        rating=1900,
        tags=["dp", "bitmasks", "sos dp"],
        notes="20-bit Sum Over Subsets (SOS) DP over masks [0, 2^20).",
        story_and_spec="""An arcane guild employs N apprentices. Apprentice i possesses a rune signature M_i (an integer bitmask) and contributes resonance P_i.

You are given Q ritual configurations. Each configuration specifies a master bitmask X: an apprentice i is compatible with X if every 1-bit in M_i is also a 1-bit in X (that is, (M_i & X) == M_i). For each configuration X, compute the sum of P_i over all compatible apprentices.

Input
The first line contains N and Q.
Each of the next N lines contains M_i and P_i.
Each of the next Q lines contains X.

Output
For each ritual configuration print the total compatible resonance on its own line.

Constraints
1 <= N, Q <= 200000
0 <= M_i, X < 2^20 (i.e. at most 1048575)
1 <= P_i <= 10^6""",
        samples=[
            "5 4\n0 10\n1 20\n2 30\n3 40\n5 50\n0\n3\n5\n7\n",
        ],
        expected_samples=["10\n100\n80\n150\n"],
        small_inputs=[
            "1 2\n0 99\n0\n1048575\n",
            gen_18(500, 600, 12, 1801),
            gen_18(1500, 1500, 20, 1802),
            gen_18(1800, 1800, 20, 1803),
        ],
        large_inputs=[
            gen_18(200000, 200000, 20, 1811),
            gen_18(200000, 200000, 20, 1812),
            gen_18(200000, 200000, 18, 1813),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int B = 20;
    const int SZ = 1 << B;
    vector<long long> f(SZ, 0);
    for (int i = 0; i < n; i++) {
        int m; long long p;
        cin >> m >> p;
        f[m] += p;
    }
    for (int b = 0; b < B; b++) {
        int bit = 1 << b;
        for (int mask = 0; mask < SZ; mask++) {
            if (mask & bit) f[mask] += f[mask ^ bit];
        }
    }
    while (q--) {
        int x;
        cin >> x;
        cout << f[x] << '\n';
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
    vector<int> m(n);
    vector<long long> p(n);
    for (int i = 0; i < n; i++) cin >> m[i] >> p[i];
    while (q--) {
        int x;
        cin >> x;
        long long s = 0;
        for (int i = 0; i < n; i++) if ((m[i] & x) == m[i]) s += p[i];
        cout << s << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_19: MERCENARY WATCH (1900) - Segment-Tree-Optimized Interval Cover DP
    # ------------------------------------------------------------------
    def gen_19(n: int, m: int, seed: int, connected: bool = True) -> str:
        r = random.Random(seed)
        lines = [f"{n} {m}"]
        segs = []
        if connected:
            cur = 1
            while cur <= n and len(segs) < m // 3:
                nxt = min(n, cur + r.randint(1, max(1, n // 15)))
                l = r.randint(max(1, cur - max(1, n // 30)), cur)
                segs.append((l, nxt, r.randint(10**6, 10**9)))
                cur = nxt + 1
        while len(segs) < m:
            l = r.randint(1, n)
            rr = min(n, l + r.randint(0, max(1, n // 5)))
            segs.append((l, rr, r.randint(1, 10**9)))
        r.shuffle(segs)
        for l, rr, w in segs:
            lines.append(f"{l} {rr} {w}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_19",
        title="Mercenary Watch",
        rating=1900,
        tags=["dp", "segment tree", "greedy", "intervals"],
        notes="Sort intervals by right endpoint; dp[r] = min(dp[r], W + min_{x in [l-1..r]} dp[x]) maintained in a point-min-update, range-min-query segment tree.",
        story_and_spec="""A fortress wall must be guarded continuously across N discrete hours numbered 1 to N. M mercenaries apply for work: mercenary i offers to guard every integer hour from L_i to R_i inclusive in exchange for wage W_i.

Select a subset of mercenaries so that every hour 1, 2, ..., N is guarded by at least one hired mercenary, minimizing the total wage paid (or report -1 if it is impossible to cover all N hours).

Input
The first line contains N and M.
Each of the next M lines contains L_i, R_i, and W_i.

Output
Print a single integer: the minimum total wage, or -1 if impossible.

Constraints
1 <= N, M <= 200000
1 <= L_i <= R_i <= N
1 <= W_i <= 10^9""",
        samples=[
            "6 4\n1 3 10\n2 5 15\n4 6 12\n1 6 30\n",
        ],
        expected_samples=["22\n"],
        small_inputs=[
            "5 2\n1 2 5\n4 5 5\n",
            gen_19(400, 600, 1901, True),
            gen_19(1500, 1800, 1902, True),
            gen_19(1500, 1500, 1903, False),
        ],
        large_inputs=[
            gen_19(180000, 200000, 1911, True),
            gen_19(200000, 200000, 1912, True),
            gen_19(200000, 200000, 1913, False),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 61);

struct SegTree {
    int sz;
    vector<long long> t;
    SegTree(int n) {
        sz = 1;
        while (sz <= n + 1) sz <<= 1;
        t.assign(2 * sz, INF);
    }
    void chmin(int idx, long long val) {
        int u = sz + idx;
        if (val >= t[u]) return;
        t[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) t[u] = min(t[2 * u], t[2 * u + 1]);
    }
    long long query(int l, int r) const { // inclusive [l..r]
        long long res = INF;
        for (l += sz, r += sz + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = min(res, t[l++]);
            if (r & 1) res = min(res, t[--r]);
        }
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    struct Seg { int l, r; long long w; };
    vector<Seg> a(m);
    for (int i = 0; i < m; i++) cin >> a[i].l >> a[i].r >> a[i].w;
    sort(a.begin(), a.end(), [](const Seg& x, const Seg& y) { return x.r < y.r; });

    SegTree st(n);
    st.chmin(0, 0);
    for (const auto& s : a) {
        long long best = st.query(s.l - 1, s.r);
        if (best < INF) st.chmin(s.r, best + s.w);
    }
    long long ans = st.query(n, n);
    cout << (ans >= INF ? -1LL : ans) << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    struct Seg { int l, r; long long w; };
    vector<Seg> a(m);
    for (int i = 0; i < m; i++) cin >> a[i].l >> a[i].r >> a[i].w;
    sort(a.begin(), a.end(), [](const Seg& x, const Seg& y) { return x.r < y.r; });

    const long long INF = (1LL << 61);
    vector<long long> dp(n + 1, INF);
    dp[0] = 0;
    for (const auto& s : a) {
        long long best = INF;
        for (int x = s.l - 1; x <= s.r; x++) best = min(best, dp[x]);
        if (best < INF) dp[s.r] = min(dp[s.r], best + s.w);
    }
    cout << (dp[n] >= INF ? -1LL : dp[n]) << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_20: VAULT RELICS (2000) - Single-Exclusion 0/1 Knapsack Queries
    # ------------------------------------------------------------------
    def gen_20(n: int, cap: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {cap} {q}"]
        for _ in range(n):
            lines.append(f"{r.randint(1, cap)} {r.randint(1, 10**9)}")
        for _ in range(q):
            lines.append(f"{r.randint(1, n)} {r.randint(0, cap)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_20",
        title="Vault Relics",
        rating=2000,
        tags=["dp", "knapsack", "prefix suffix"],
        notes="Precompute prefix knapsack dpL[i][w] and suffix knapsack dpR[i][w] in O(N*C), then answer each (banned k, cap c) query in O(C) via max_{0<=w<=c} (dpL[k-1][w] + dpR[k+1][c-w]).",
        story_and_spec="""A royal vault holds N relics numbered 1 to N. Relic i has weight W_i and value V_i (at most one copy of each relic may be taken). The maximum possible satchel capacity is C.

You must answer Q contingency queries. Query j specifies a forbidden relic K_j and a satchel capacity B_j: if relic K_j may NOT be taken, what is the maximum total value of a subset of the remaining N - 1 relics whose total weight is at most B_j?

Input
The first line contains N, C, and Q.
Each of the next N lines contains W_i and V_i.
Each of the next Q lines contains K_j and B_j.

Output
For each query print the maximum achievable total value on its own line.

Constraints
2 <= N <= 2500
1 <= C <= 2000
1 <= Q <= 120000
1 <= W_i <= C
1 <= V_i <= 10^9
1 <= K_j <= N
0 <= B_j <= C""",
        samples=[
            "4 5 3\n2 10\n3 15\n2 12\n4 20\n2 5\n4 4\n1 2\n",
        ],
        expected_samples=["22\n22\n12\n"],
        small_inputs=[
            "2 3 2\n2 5\n3 8\n1 3\n2 1\n",
            gen_20(80, 120, 120, 2001),
            gen_20(140, 160, 140, 2002),
            gen_20(160, 180, 150, 2003),
        ],
        large_inputs=[
            gen_20(2200, 1800, 100000, 2011),
            gen_20(2500, 2000, 120000, 2012),
            gen_20(2500, 2000, 120000, 2013),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, C, q;
    cin >> n >> C >> q;
    vector<int> w(n + 1);
    vector<long long> v(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i] >> v[i];

    int stride = C + 1;
    vector<long long> pref((n + 2) * stride, 0), suff((n + 2) * stride, 0);

    for (int i = 1; i <= n; i++) {
        const long long* prv = &pref[(i - 1) * stride];
        long long* cur = &pref[i * stride];
        int wi = w[i];
        long long vi = v[i];
        memcpy(cur, prv, sizeof(long long) * stride);
        for (int c = wi; c <= C; c++) {
            cur[c] = max(cur[c], prv[c - wi] + vi);
        }
    }
    for (int i = n; i >= 1; i--) {
        const long long* nxt = &suff[(i + 1) * stride];
        long long* cur = &suff[i * stride];
        int wi = w[i];
        long long vi = v[i];
        memcpy(cur, nxt, sizeof(long long) * stride);
        for (int c = wi; c <= C; c++) {
            cur[c] = max(cur[c], nxt[c - wi] + vi);
        }
    }

    while (q--) {
        int k, b;
        cin >> k >> b;
        const long long* L = &pref[(k - 1) * stride];
        const long long* R = &suff[(k + 1) * stride];
        long long best = 0;
        for (int x = 0; x <= b; x++) {
            best = max(best, L[x] + R[b - x]);
        }
        cout << best << '\n';
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
    int n, C, q;
    cin >> n >> C >> q;
    vector<int> w(n + 1);
    vector<long long> v(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i] >> v[i];
    vector<long long> dp(C + 1);
    while (q--) {
        int k, b;
        cin >> k >> b;
        fill(dp.begin(), dp.begin() + b + 1, 0LL);
        for (int i = 1; i <= n; i++) if (i != k && w[i] <= b) {
            for (int c = b; c >= w[i]; c--) dp[c] = max(dp[c], dp[c - w[i]] + v[i]);
        }
        cout << dp[b] << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_04())
