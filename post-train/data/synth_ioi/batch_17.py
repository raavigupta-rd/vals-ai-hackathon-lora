"""Batch 17 (synth_81 .. synth_85): Grandmaster Wave I (CF 2600-2800) — Weighted Shifting-Interval Slope Trick, Dynamic Tree MWIS (HLD Tropical Matrix SegTree), Harbingers Ancestor Undo Li Chao Tree, WQS Aliens + CHT Exact-K Squared Partition, Dynamic Segment Tree Merge."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def _rand_tree(n: int, r: random.Random, kind: str = "random") -> list[tuple[int, int]]:
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    i1 = perm.index(1)
    perm[0], perm[i1] = perm[i1], perm[0]
    es = []
    for i in range(1, n):
        if kind == "chain":
            p = perm[i - 1]
        elif kind == "caterpillar":
            p = perm[max(0, i - r.randint(1, 3))]
        else:
            p = perm[r.randint(0, i - 1)]
        es.append((p, perm[i]))
    return es


def make_batch_17() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_81: TERRACED AQUEDUCTS (2700) - Weighted Shifting-Interval Slope Trick
    # ------------------------------------------------------------------
    def gen_81(n: int, amax: int, dmax: int, wmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(-amax, amax) for _ in range(n)]
        w = [r.randint(1, wmax) for _ in range(n)]
        lines = [str(n), " ".join(map(str, a)), " ".join(map(str, w))]
        for _ in range(n - 1):
            lo = r.randint(-dmax, dmax)
            hi = r.randint(lo, min(dmax, lo + r.randint(0, max(1, dmax))))
            lines.append(f"{lo} {hi}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_81",
        title="Terraced Aqueducts",
        rating=2700,
        tags=["dp", "slope trick", "convexity", "data structures", "heaps"],
        notes="Weighted Slope Trick with two run-length-encoded heaps (coord, slope_delta) and lazy offsets (shift_L, shift_R) in O(N log N); verified on small via independent monotone-deque coordinate DP.",
        story_and_spec="""An imperial engineer is grading N aqueduct piers in sequence. Pier i currently sits at integer elevation A_i, and raising or lowering pier i to integer elevation B_i costs W_i * |B_i - A_i| gold per unit of vertical change.

Hydraulic flow requires that for every consecutive pair of piers (i - 1, i) with 2 <= i <= N, the elevation step satisfies:
L_i <= B_i - B_{i-1} <= R_i.

Choose integer elevations B_1, B_2, ..., B_N (which may be negative, zero, or positive) satisfying all N - 1 step bounds to minimize the total earthwork cost sum_{i=1..N} W_i * |B_i - A_i|.

Input
The first line contains N.
The second line contains A_1 ... A_N.
The third line contains W_1 ... W_N.
Each of the next N - 1 lines contains two integers L_i and R_i (for i = 2, ..., N).

Output
Print a single integer: the minimum possible total cost.

Constraints
1 <= N <= 180000
-10^7 <= A_i <= 10^7
1 <= W_i <= 10^4
-10^4 <= L_i <= R_i <= 10^4""",
        samples=[
            "4\n10 2 9 1\n3 2 4 5\n-2 1\n0 3\n-4 -1\n",
            "3\n0 100 0\n1 10 1\n-5 5\n-5 5\n",
        ],
        small_inputs=[
            "1\n-42\n99\n",
            gen_81(120, 400, 15, 500, 8101),
            gen_81(260, 2500, 20, 10000, 8102),
            gen_81(220, 1500, 0, 10000, 8103),
        ],
        large_inputs=[
            # large[0] keeps coordinate span ~24,000 so O(N*V) deque-DP uses <1 MB RAM and cleanly TLEs!
            gen_81(180000, 10000, 10, 10000, 8111),
            gen_81(180000, 10**7, 10000, 10000, 8112),
            gen_81(180000, 10**7, 500, 10000, 8113),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n), w(n), lo(n, 0), hi(n, 0);
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < n; i++) cin >> w[i];
    for (int i = 1; i < n; i++) cin >> lo[i] >> hi[i];

    using P = pair<long long, long long>; // (raw_coord, slope_weight)
    priority_queue<P> pqL;
    priority_queue<P, vector<P>, greater<P>> pqR;
    long long shL = 0, shR = 0;
    long long min_val = 0;

    pqL.push({a[0], w[0]});
    pqR.push({a[0], w[0]});

    for (int i = 1; i < n; i++) {
        shL += lo[i];
        shR += hi[i];
        long long ai = a[i], wi = w[i];
        long long l_star = pqL.top().first + shL;
        long long r_star = pqR.top().first + shR;

        if (ai < l_star) {
            long long rem = wi;
            while (rem > 0 && !pqL.empty() && pqL.top().first + shL > ai) {
                auto [xr, cnt] = pqL.top(); pqL.pop();
                long long x = xr + shL;
                long long take = min(rem, cnt);
                min_val += take * (x - ai);
                pqR.push({x - shR, take});
                if (cnt > take) pqL.push({xr, cnt - take});
                rem -= take;
            }
            // (wi - rem) units moved L->R (each added 2 at ai to L), plus rem units in middle (1 to L, 1 to R)
            pqL.push({ai - shL, 2 * wi - rem});
            if (rem > 0) pqR.push({ai - shR, rem});
        } else if (ai > r_star) {
            long long rem = wi;
            while (rem > 0 && !pqR.empty() && pqR.top().first + shR < ai) {
                auto [xr, cnt] = pqR.top(); pqR.pop();
                long long x = xr + shR;
                long long take = min(rem, cnt);
                min_val += take * (ai - x);
                pqL.push({x - shL, take});
                if (cnt > take) pqR.push({xr, cnt - take});
                rem -= take;
            }
            pqR.push({ai - shR, 2 * wi - rem});
            if (rem > 0) pqL.push({ai - shL, rem});
        } else {
            pqL.push({ai - shL, wi});
            pqR.push({ai - shR, wi});
        }
    }

    cout << min_val << '\n';
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
    vector<long long> a(n), w(n), lo(n, 0), hi(n, 0);
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < n; i++) cin >> w[i];
    long long max_step = 0;
    for (int i = 1; i < n; i++) {
        cin >> lo[i] >> hi[i];
        max_step += max(llabs(lo[i]), llabs(hi[i]));
    }
    if (n == 1) {
        cout << 0 << '\n';
        return 0;
    }

    // Independent O(N * V) monotone-deque DP over feasible integer range [mnB, mxB]
    // Note: every optimal B_i lies in [min(A) - sum|step|, max(A) + sum|step|],
    // or more tightly within [min_j (A_j - dist), max_j (A_j + dist)].
    // Even tighter: any B outside [min(A) - max_local, max(A) + max_local]? Actually if all B_i
    // can be shifted toward A, let's bound reachable [L_reach, R_reach] around A!
    // Wait: if B_1 in [min_j(A_j - pref_R[j]), max_j(A_j - pref_L[j])]? Not if pref_L == pref_R!
    // Exact safe box: let prefL[i] = sum_{2..i} lo, prefR[i] = sum_{2..i} hi.
    // Optimal B_i always equals A_j + shift for some j, where shift in [prefL[i]-prefR[j], prefR[i]-prefL[j]]!
    vector<long long> pL(n, 0), pR(n, 0);
    for (int i = 1; i < n; i++) { pL[i] = pL[i - 1] + lo[i]; pR[i] = pR[i - 1] + hi[i]; }
    long long mnB = LLONG_MAX, mxB = LLONG_MIN;
    if (n <= 400) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mnB = min(mnB, a[j] + pL[i] - pR[j]);
                mxB = max(mxB, a[j] + pR[i] - pL[j]);
            }
        }
    } else {
        long long amin = *min_element(a.begin(), a.end());
        long long amax = *max_element(a.begin(), a.end());
        mnB = amin - 5000;
        mxB = amax + 5000;
    }

    int V = (int)(mxB - mnB + 1);
    const long long INF = (1LL << 61);
    vector<long long> dp(V), ndp(V, INF);
    for (int x = 0; x < V; x++) dp[x] = w[0] * llabs((mnB + x) - a[0]);

    deque<int> dq;
    for (int i = 1; i < n; i++) {
        fill(ndp.begin(), ndp.end(), INF);
        dq.clear();
        // For target y = mnB + x, prev y' = y - step in [y - hi[i], y - lo[i]], i.e. idx' in [x - hi[i], x - lo[i]]
        long long Loff = -hi[i], Roff = -lo[i];
        int ptr = max(0LL, Loff);
        for (int x = 0; x < V; x++) {
            long long ql = x + Loff, qr = x + Roff;
            while (ptr <= qr && ptr < V) {
                if (ptr >= 0) {
                    while (!dq.empty() && dp[dq.back()] >= dp[ptr]) dq.pop_back();
                    dq.push_back(ptr);
                }
                ptr++;
            }
            while (!dq.empty() && dq.front() < ql) dq.pop_front();
            if (!dq.empty() && dp[dq.front()] < INF) {
                ndp[x] = dp[dq.front()] + w[i] * llabs((mnB + x) - a[i]);
            }
        }
        dp.swap(ndp);
    }
    cout << *min_element(dp.begin(), dp.end()) << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_82: GARRISON REASSIGNMENTS (2700) - Dynamic Tree MWIS (HLD Matrix SegTree)
    # ------------------------------------------------------------------
    def gen_82(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        a = [r.randint(0, 10**9) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, a))] + [f"{u} {v}" for u, v in es]
        for _ in range(q):
            lines.append(f"{r.randint(1, n)} {r.randint(0, 10**9)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_82",
        title="Garrison Reassignments",
        rating=2700,
        tags=["trees", "heavy-light decomposition", "dynamic dp", "matrices", "segment tree"],
        notes="Dynamic Tree DP (DDP): decompose tree into heavy chains, maintain light-child sums g0[u], g1[u] as 2x2 (max, +) tropical matrices in a Segment Tree; each point weight update walks O(log N) chains in O(log^2 N).",
        story_and_spec="""A frontier kingdom has N fortresses numbered 1 to N connected by N - 1 bidirectional roads forming a tree. Initially fortress i holds garrison strength A_i.

Two fortresses connected directly by a road cannot simultaneously mobilize without leaving their shared supply road exposed, so any mobilized subset of fortresses must form an INDEPENDENT SET (no two mobilized fortresses share a road).

There are Q sequential reassignments. Each reassignment gives a fortress U and a new garrison strength X, permanently setting A_U <- X. After EVERY reassignment, report the maximum possible total garrison strength of a valid mobilized subset of fortresses.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next N - 1 lines contains two fortresses connected by a road.
Each of the next Q lines contains U and X.

Output
Print Q lines: after each reassignment, print the maximum independent-set strength of the tree.

Constraints
1 <= N, Q <= 120000
0 <= A_i, X <= 10^9
1 <= U <= N""",
        samples=[
            "5 4\n10 20 30 40 50\n1 2\n2 3\n2 4\n4 5\n2 100\n4 5\n5 1\n2 10\n",
        ],
        small_inputs=[
            "1 2\n5\n1 0\n1 99\n",
            gen_82(300, 400, 8201),
            gen_82(1400, 1400, 8202, "caterpillar"),
        ],
        large_inputs=[
            gen_82(120000, 120000, 8211, "caterpillar"),
            gen_82(120000, 120000, 8212, "chain"),
            gen_82(120000, 120000, 8213, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long NEG_INF = -(1LL << 60);

struct Mat {
    long long m[2][2];
    Mat() {
        m[0][0] = m[1][1] = 0;
        m[0][1] = m[1][0] = NEG_INF;
    }
};

static inline Mat mul(const Mat& A, const Mat& B) {
    Mat C;
    C.m[0][0] = max(A.m[0][0] + B.m[0][0], A.m[0][1] + B.m[1][0]);
    C.m[0][1] = max(A.m[0][0] + B.m[0][1], A.m[0][1] + B.m[1][1]);
    C.m[1][0] = max(A.m[1][0] + B.m[0][0], A.m[1][1] + B.m[1][0]);
    C.m[1][1] = max(A.m[1][0] + B.m[0][1], A.m[1][1] + B.m[1][1]);
    return C;
}

static inline Mat make_node_mat(long long g0, long long g1) {
    Mat M;
    M.m[0][0] = g0; M.m[0][1] = g0;
    M.m[1][0] = g1; M.m[1][1] = NEG_INF;
    return M;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0), ord = {1};
    ord.reserve(n);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ord.push_back(v);
        }
    }
    for (int i = n - 1; i >= 1; i--) {
        int u = ord[i];
        sz[par[u]] += sz[u];
        if (sz[u] > sz[heavy[par[u]]]) heavy[par[u]] = u;
    }

    vector<int> head(n + 1, 0), tail(n + 1, 0), pos(n + 1, 0), inv_pos(n + 1, 0);
    int timer = 0;
    {
        vector<pair<int, int>> st = {{1, 1}};
        while (!st.empty()) {
            auto [u, h] = st.back(); st.pop_back();
            int last = u;
            for (int x = u; x != 0; x = heavy[x]) {
                head[x] = h;
                pos[x] = ++timer;
                inv_pos[timer] = x;
                last = x;
                for (int v : adj[x]) if (v != par[x] && v != heavy[x]) st.push_back({v, v});
            }
            tail[h] = last;
        }
    }

    vector<long long> dp0(n + 1, 0), dp1(n + 1, 0), g0(n + 1, 0), g1(n + 1, 0);
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        g1[u] = a[u];
        for (int v : adj[u]) if (v != par[u] && v != heavy[u]) {
            g0[u] += max(dp0[v], dp1[v]);
            g1[u] += dp0[v];
        }
        dp0[u] = g0[u];
        dp1[u] = g1[u];
        if (heavy[u]) {
            dp0[u] += max(dp0[heavy[u]], dp1[heavy[u]]);
            dp1[u] += dp0[heavy[u]];
        }
    }

    int S = 1;
    while (S < n) S <<= 1;
    vector<Mat> seg(2 * S);
    for (int i = 1; i <= n; i++) {
        int u = inv_pos[i];
        seg[S + i - 1] = make_node_mat(g0[u], g1[u]);
    }
    for (int i = S - 1; i >= 1; i--) seg[i] = mul(seg[2 * i], seg[2 * i + 1]);

    auto seg_set = [&](int idx, const Mat& M) {
        int u = S + idx - 1;
        seg[u] = M;
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = mul(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_query = [&](int l, int r) {
        Mat L, R;
        for (l += S - 1, r += S; l < r; l >>= 1, r >>= 1) {
            if (l & 1) L = mul(L, seg[l++]);
            if (r & 1) R = mul(seg[--r], R);
        }
        return mul(L, R);
    };
    auto get_chain_dp = [&](int h) -> pair<long long, long long> {
        Mat P = seg_query(pos[h], pos[tail[h]]);
        // Applied to base column [0, NEG_INF]^T -> [P.m[0][0], P.m[1][0]]
        return {P.m[0][0], P.m[1][0]};
    };

    while (q--) {
        int u; long long nx;
        cin >> u >> nx;
        g1[u] += nx - a[u];
        a[u] = nx;

        while (u != 0) {
            int h = head[u];
            int p = par[h];
            pair<long long, long long> old_dp = {0, 0};
            if (p != 0) old_dp = {dp0[h], dp1[h]};

            seg_set(pos[u], make_node_mat(g0[u], g1[u]));
            auto new_dp = get_chain_dp(h);
            dp0[h] = new_dp.first;
            dp1[h] = new_dp.second;

            if (p != 0) {
                g0[p] += max(new_dp.first, new_dp.second) - max(old_dp.first, old_dp.second);
                g1[p] += new_dp.first - old_dp.first;
            }
            u = p;
        }
        cout << max(dp0[1], dp1[1]) << '\n';
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
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1, 0), ord = {1};
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ord.push_back(v);
        }
    }
    vector<long long> dp0(n + 1), dp1(n + 1);
    while (q--) {
        int u; long long x;
        cin >> u >> x;
        a[u] = x;
        for (int i = n - 1; i >= 0; i--) {
            int v = ord[i];
            long long s0 = 0, s1 = a[v];
            for (int nb : adj[v]) if (nb != par[v]) {
                s0 += max(dp0[nb], dp1[nb]);
                s1 += dp0[nb];
            }
            dp0[v] = s0;
            dp1[v] = s1;
        }
        cout << max(dp0[1], dp1[1]) << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_83: HARBOR BEACONS (2700) - CEOI Harbingers (Tree Ancestor Undo Li Chao)
    # ------------------------------------------------------------------
    def gen_83(n: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        lines = [str(n)]
        for u, v in es:
            lines.append(f"{u} {v} {r.randint(1, 4000)}")
        for _ in range(2, n + 1):
            lines.append(f"{r.randint(0, 10**9)} {r.randint(1, 10**9)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_83",
        title="Harbor Beacons",
        rating=2700,
        tags=["dp", "trees", "convex hull trick", "li chao tree", "rollback"],
        notes="CEOI Harbingers: dp[u] = S_u + D_u * P_u + min_{a in anc(u)} (-D_a * P_u + dp[a]); maintain an Undo-Stack Li Chao Segment Tree over compressed P_u along the DFS ancestor stack in O(N log N).",
        story_and_spec="""An archipelago kingdom has N coastal harbors numbered 1 to N connected by N - 1 bidirectional sea lanes forming a tree, with the royal capital at harbor 1. Sea lane i has length W_i kilometers.

Each harbor u (for 2 <= u <= N) has a courier boat that requires S_u minutes of launch preparation and then travels toward the capital at a pace of P_u minutes per kilometer.

When a dispatch originates at harbor u (u >= 2) bound for harbor 1 along the unique simple path to 1, whenever the current courier arrives at any intermediate harbor a on that path, you may EITHER stay on the current boat OR transfer the dispatch to harbor a's courier boat (paying S_a preparation minutes and continuing at pace P_a). You may transfer as many times as you wish.

Compute for every harbor u = 2, 3, ..., N the minimum total time in minutes for a dispatch from harbor u to reach harbor 1.

Input
The first line contains N.
Each of the next N - 1 lines contains U_i, V_i, and W_i describing a sea lane.
Each of the next N - 1 lines (for u = 2, 3, ..., N in order) contains two integers S_u and P_u.

Output
Print N - 1 space-separated integers: the minimum dispatch time from harbors 2, 3, ..., N to harbor 1.

Constraints
2 <= N <= 150000
1 <= U_i, V_i <= N
1 <= W_i <= 4000
0 <= S_u <= 10^9
1 <= P_u <= 10^9""",
        samples=[
            "5\n1 2 10\n2 3 5\n2 4 8\n4 5 12\n20 10\n5 30\n50 4\n10 25\n",
        ],
        small_inputs=[
            "2\n1 2 100\n7 3\n",
            gen_83(300, 8301),
            gen_83(2000, 8302, "chain"),
        ],
        large_inputs=[
            gen_83(150000, 8311, "chain"),
            gen_83(150000, 8312, "caterpillar"),
            gen_83(150000, 8313, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct Line {
    long long m, b;
    bool empty = true;
    inline __int128 eval(long long x) const {
        if (empty) return (__int128)1 << 120;
        return (__int128)m * x + b;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<long long> S(n + 1, 0), P(n + 1, 0), xs;
    xs.reserve(n);
    for (int u = 2; u <= n; u++) {
        cin >> S[u] >> P[u];
        xs.push_back(P[u]);
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());
    int M = (int)xs.size();

    vector<Line> seg(4 * M + 5);
    vector<pair<int, Line>> undo;
    undo.reserve(n * 20);

    auto insert_line = [&](Line nw) {
        int u = 1, l = 0, r = M - 1;
        while (true) {
            int mid = (l + r) >> 1;
            bool left_better = nw.eval(xs[l]) < seg[u].eval(xs[l]);
            bool mid_better = nw.eval(xs[mid]) < seg[u].eval(xs[mid]);
            if (mid_better) {
                undo.push_back({u, seg[u]});
                swap(seg[u], nw);
            }
            if (l == r || nw.empty) break;
            if (left_better != mid_better) {
                u = u << 1;
                r = mid;
            } else {
                u = u << 1 | 1;
                l = mid + 1;
            }
        }
    };

    auto query_x = [&](int idx) -> long long {
        long long x = xs[idx];
        __int128 res = (__int128)1 << 120;
        int u = 1, l = 0, r = M - 1;
        while (true) {
            res = min(res, seg[u].eval(x));
            if (l == r) break;
            int mid = (l + r) >> 1;
            if (idx <= mid) { u = u << 1; r = mid; }
            else { u = u << 1 | 1; l = mid + 1; }
        }
        return (long long)res;
    };

    vector<long long> dist(n + 1, 0), dp(n + 1, 0);
    vector<int> par(n + 1, 0);
    struct Frame { int u, idx; size_t snap; };
    vector<Frame> st = {{1, 0, 0}};

    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.idx == 0) {
            f.snap = undo.size();
            if (u != 1) {
                int qi = (int)(lower_bound(xs.begin(), xs.end(), P[u]) - xs.begin());
                dp[u] = S[u] + dist[u] * P[u] + query_x(qi);
            }
            insert_line({-dist[u], dp[u], false});
        }
        bool pushed = false;
        while (f.idx < (int)adj[u].size()) {
            auto [v, w] = adj[u][f.idx++];
            if (v == par[u]) continue;
            par[v] = u;
            dist[v] = dist[u] + w;
            st.push_back({v, 0, 0});
            pushed = true;
            break;
        }
        if (!pushed) {
            while (undo.size() > f.snap) {
                auto [node, old_l] = undo.back();
                undo.pop_back();
                seg[node] = old_l;
            }
            st.pop_back();
        }
    }

    for (int u = 2; u <= n; u++) {
        cout << dp[u] << (u == n ? '\n' : ' ');
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
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<long long> S(n + 1, 0), P(n + 1, 0), dist(n + 1, 0), dp(n + 1, 0);
    for (int u = 2; u <= n; u++) cin >> S[u] >> P[u];
    vector<int> par(n + 1, 0), ord = {1};
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, w] : adj[u]) if (v != par[u]) {
            par[v] = u;
            dist[v] = dist[u] + w;
            ord.push_back(v);
        }
    }
    for (int i = 1; i < n; i++) {
        int u = ord[i];
        __int128 best = (__int128)1 << 120;
        for (int a = par[u]; a != 0; a = par[a]) {
            __int128 cand = (__int128)dp[a] + (__int128)(dist[u] - dist[a]) * P[u] + S[u];
            if (cand < best) best = cand;
        }
        dp[u] = (long long)best;
    }
    for (int u = 2; u <= n; u++) cout << dp[u] << (u == n ? '\n' : ' ');
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_84: SEGMENTED CONVOYS (2800) - WQS Aliens + Monotonic CHT
    # (Even hidden_small has N=3500, K=1500 so naive O(N^2 K) DP scores 0!)
    # ------------------------------------------------------------------
    def gen_84(n: int, k: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        return f"{n} {k}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_84",
        title="Segmented Convoys",
        rating=2800,
        tags=["dp", "wqs binary search", "aliens trick", "convex hull trick"],
        notes="Calibrated so naive O(N^2 K) DP TLEs even on small (N=3500, K=1500): 40-pt subtask requires O(NK) CHT/D&C DP, and 100-pt requires WQS Binary Search + O(N) Monotonic CHT with BIG=(N+2) lexicographic tie-encoding.",
        story_and_spec="""A logistics fleet must divide a sequence of N crates with positive weights A_1, A_2, ..., A_N (preserving their order) into EXACTLY K non-empty contiguous convoys.

If convoy t carries crates L_t, L_t + 1, ..., R_t with total weight S_t = A_{L_t} + ... + A_{R_t}, road wear imposes a quadratic surcharge of (S_t)^2.

Find the minimum possible total surcharge sum_{t=1..K} (S_t)^2 across all valid partitions into K non-empty contiguous convoys.

Input
The first line contains N and K.
The second line contains A_1 ... A_N.

Output
Print a single integer: the minimum possible sum of squared convoy weights.

Constraints
1 <= K <= N <= 150000
1 <= A_i <= 2000""",
        samples=[
            "6 3\n3 1 4 1 5 2\n",
            "4 4\n7 2 5 3\n",
        ],
        small_inputs=[
            "5 1\n2 3 4 5 6\n",
            gen_84(400, 120, 2000, 8401),
            # N=3500, K=1500: O(N^2 K) ~ 9.2e9 ops -> TLE even on small; O(NK) CHT = 5.2e6 ops -> 15ms!
            gen_84(3500, 1500, 2000, 8402),
            gen_84(3000, 800, 50, 8403),
        ],
        large_inputs=[
            gen_84(150000, 50000, 2000, 8411),
            gen_84(150000, 12000, 2000, 8412),
            gen_84(150000, 90000, 20, 8413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, K;
    cin >> n >> K;
    vector<long long> P(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        P[i] = P[i - 1] + x;
    }
    if (K == n) {
        long long s = 0;
        for (int i = 1; i <= n; i++) {
            long long d = P[i] - P[i - 1];
            s += d * d;
        }
        cout << s << '\n';
        return 0;
    }

    const __int128 BIG = n + 2;
    vector<__int128> dp(n + 1, 0);
    vector<int> dq(n + 2);

    // Solve penalized CHT minimizing BIG * (sum S_t^2 + lam * k) + k
    auto solve_pen = [&](long long lam) -> pair<long long, int> {
        const __int128 pen = (__int128)lam * BIG + 1;
        auto M_of = [&](int j) -> __int128 { return -2 * BIG * P[j]; };
        auto B_of = [&](int j) -> __int128 { return dp[j] + BIG * P[j] * P[j]; };
        auto eval = [&](int j, long long x) -> __int128 { return M_of(j) * x + B_of(j); };

        dp[0] = 0;
        int head = 0, tail = 0;
        dq[tail++] = 0;

        for (int i = 1; i <= n; i++) {
            long long x = P[i];
            while (tail - head >= 2 && eval(dq[head + 1], x) <= eval(dq[head], x)) head++;
            int best_j = dq[head];
            dp[i] = eval(best_j, x) + BIG * x * x + pen;

            // Slopes M_of(j) = -2*BIG*P[j] are strictly decreasing; pop redundant back lines
            __int128 m3 = M_of(i), b3 = B_of(i);
            while (tail - head >= 2) {
                int j1 = dq[tail - 2], j2 = dq[tail - 1];
                __int128 m1 = M_of(j1), b1 = B_of(j1);
                __int128 m2 = M_of(j2), b2 = B_of(j2);
                // Intersection(j1, j2) = (b2 - b1)/(m1 - m2) >= (b3 - b2)/(m2 - m3)
                // Multiply by positive (m1 - m2)*(m2 - m3):
                // Wait: P[N] <= 3e8, BIG <= 1.5e5 -> b <= 1.5e22, m <= 9e13, b*m <= 1.35e36 < INT128_MAX (1.7e38)
                if ((b2 - b1) * (m2 - m3) >= (b3 - b2) * (m1 - m2)) tail--;
                else break;
            }
            dq[tail++] = i;
        }

        int cnt = (int)(dp[n] % BIG);
        long long f_plus_lam_k = (long long)(dp[n] / BIG);
        return {f_plus_lam_k, cnt};
    };

    long long lo = 0, hi = P[n] * P[n] + 1;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (solve_pen(mid).second <= K) hi = mid;
        else lo = mid + 1;
    }
    auto [f_pen, _] = solve_pen(lo);
    cout << (f_pen - (__int128)lo * K > 0 ? (long long)((__int128)f_pen - (__int128)lo * K) : 0LL) << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

// 40-pt subtask solver: O(N * K) Monotonic-Deque Convex Hull Trick DP (O(N) memory)
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, K;
    cin >> n >> K;
    vector<long long> P(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        P[i] = P[i - 1] + x;
    }
    const long long INF = (1LL << 61);
    vector<long long> prv(n + 1, INF), cur(n + 1, INF);
    prv[0] = 0;
    vector<int> dq(n + 2);

    for (int t = 1; t <= K; t++) {
        auto M_of = [&](int j) { return -2LL * P[j]; };
        auto B_of = [&](int j) { return prv[j] + P[j] * P[j]; };
        auto eval = [&](int j, long long x) { return M_of(j) * x + B_of(j); };

        int head = 0, tail = 0;
        dq[tail++] = t - 1;
        for (int i = t; i <= n; i++) {
            long long x = P[i];
            while (tail - head >= 2 && eval(dq[head + 1], x) <= eval(dq[head], x)) head++;
            cur[i] = eval(dq[head], x) + x * x;

            if (prv[i] < INF) {
                long long m3 = M_of(i), b3 = B_of(i);
                while (tail - head >= 2) {
                    int j1 = dq[tail - 2], j2 = dq[tail - 1];
                    __int128 lhs = (__int128)(B_of(j2) - B_of(j1)) * (M_of(j2) - m3);
                    __int128 rhs = (__int128)(b3 - B_of(j2)) * (M_of(j1) - M_of(j2));
                    if (lhs >= rhs) tail--;
                    else break;
                }
                dq[tail++] = i;
            }
        }
        fill(prv.begin(), prv.begin() + t, INF);
        for (int i = t; i <= n; i++) prv[i] = cur[i];
    }
    cout << prv[n] << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_85: SUBTREE COLOR DOMINANCE (2600) - Dynamic Segment Tree Merge
    # ------------------------------------------------------------------
    def gen_85(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        cols = [r.randint(1, n) for _ in range(n)]
        ws = [r.randint(1, 10**8) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, cols)), " ".join(map(str, ws))]
        lines += [f"{u} {v}" for u, v in es]
        for _ in range(q):
            u = 1 if (kind == "chain" and r.random() < 0.4) else r.randint(1, n)
            if kind == "chain" and r.random() < 0.7:
                u = es[r.randint(0, max(0, n // 5))][0]
            l, rr = r.randint(1, n), r.randint(1, n)
            if l > rr:
                l, rr = rr, l
            lines.append(f"{u} {l} {rr}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_85",
        title="Subtree Color Dominance",
        rating=2600,
        tags=["trees", "segment tree", "segment tree merge", "offline queries"],
        notes="Post-order Dynamic Segment Tree Merge maintaining leaf color sum S(u, c) and internal range sum of S(u, c)^2 mod (10^9+7); answers subtree color-interval queries in O((N + Q) log N).",
        story_and_spec="""A botanical archive organizes N specimen vaults numbered 1 to N into a rooted tree with vault 1 as the root. Vault u stores specimens of family C_u (1 <= C_u <= N) with rarity score W_u.

For any vault u and family c, define S(u, c) to be the sum of W_v over all vaults v in the subtree of u that belong to family c (if no vault in u's subtree belongs to family c, then S(u, c) = 0).

Answer Q queries. Each query gives a subtree root U and a family range [L, R] (1 <= L <= R <= N): compute
sum_{c=L..R} (S(U, c))^2 modulo 10^9 + 7.

Input
The first line contains N and Q.
The second line contains C_1 ... C_N.
The third line contains W_1 ... W_N.
Each of the next N - 1 lines contains two vaults connected by a corridor.
Each of the next Q lines contains U, L, and R.

Output
For each query print the sum of squared family totals modulo 10^9 + 7 on its own line.

Constraints
1 <= N, Q <= 160000
1 <= C_u <= N
1 <= W_u <= 10^8
1 <= U <= N
1 <= L <= R <= N""",
        samples=[
            "5 4\n2 1 2 3 1\n3 4 5 2 6\n1 2\n1 3\n2 4\n2 5\n1 1 3\n2 1 2\n1 2 2\n3 1 3\n",
        ],
        small_inputs=[
            "1 1\n1\n10\n1 1 1\n",
            gen_85(300, 400, 8501),
            gen_85(1800, 1800, 8502, "chain"),
        ],
        large_inputs=[
            gen_85(160000, 160000, 8511, "chain"),
            gen_85(160000, 160000, 8512, "caterpillar"),
            gen_85(160000, 160000, 8513, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 1'000'000'007LL;

struct Node {
    int ls = 0, rs = 0;
    long long sw = 0, sq = 0;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> col(n + 1);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    for (int i = 1; i <= n; i++) { cin >> w[i]; w[i] %= MOD; }

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    struct Qry { int l, r, id; };
    vector<vector<Qry>> qs(n + 1);
    for (int i = 0; i < q; i++) {
        int u, l, r; cin >> u >> l >> r;
        qs[u].push_back({l, r, i});
    }

    vector<int> par(n + 1, 0), ord = {1};
    ord.reserve(n);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ord.push_back(v);
        }
    }

    vector<Node> tr;
    tr.reserve(n * 20 + 5);
    tr.push_back({}); // 0 = null

    auto new_node = [&]() -> int {
        tr.push_back({});
        return (int)tr.size() - 1;
    };

    function<int(int, int, int, long long)> make_single =
        [&](int l, int r, int pos, long long val) -> int {
            int u = new_node();
            if (l == r) {
                tr[u].sw = val;
                tr[u].sq = val * val % MOD;
                return u;
            }
            int mid = (l + r) >> 1;
            if (pos <= mid) tr[u].ls = make_single(l, mid, pos, val);
            else tr[u].rs = make_single(mid + 1, r, pos, val);
            tr[u].sq = (tr[tr[u].ls].sq + tr[tr[u].rs].sq) % MOD;
            return u;
        };

    function<int(int, int, int, int)> merge_tr =
        [&](int a, int b, int l, int r) -> int {
            if (!a || !b) return a ? a : b;
            if (l == r) {
                tr[a].sw = (tr[a].sw + tr[b].sw) % MOD;
                tr[a].sq = tr[a].sw * tr[a].sw % MOD;
                return a;
            }
            int mid = (l + r) >> 1;
            tr[a].ls = merge_tr(tr[a].ls, tr[b].ls, l, mid);
            tr[a].rs = merge_tr(tr[a].rs, tr[b].rs, mid + 1, r);
            tr[a].sq = (tr[tr[a].ls].sq + tr[tr[a].rs].sq) % MOD;
            return a;
        };

    function<long long(int, int, int, int, int)> query_tr =
        [&](int u, int l, int r, int ql, int qr) -> long long {
            if (!u || ql <= l && r <= qr) return tr[u].sq;
            int mid = (l + r) >> 1;
            long long res = 0;
            if (ql <= mid) res += query_tr(tr[u].ls, l, mid, ql, qr);
            if (qr > mid) res += query_tr(tr[u].rs, mid + 1, r, ql, qr);
            return res % MOD;
        };

    vector<int> root(n + 1, 0);
    vector<long long> ans(q, 0);

    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        root[u] = make_single(1, n, col[u], w[u]);
        for (int v : adj[u]) if (v != par[u]) {
            root[u] = merge_tr(root[u], root[v], 1, n);
        }
        for (const auto& qr : qs[u]) {
            ans[qr.id] = query_tr(root[u], 1, n, qr.l, qr.r);
        }
    }

    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 1'000'000'007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> col(n + 1);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    for (int i = 1; i <= n; i++) cin >> w[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> tin(n + 1), tout(n + 1), flat(n + 1), par(n + 1, 0);
    int timer = 0;
    vector<pair<int, int>> st = {{1, 0}};
    while (!st.empty()) {
        auto& [u, idx] = st.back();
        if (idx == 0) {
            tin[u] = ++timer;
            flat[timer] = u;
        }
        if (idx < (int)adj[u].size()) {
            int v = adj[u][idx++];
            if (v == par[u]) continue;
            par[v] = u;
            st.push_back({v, 0});
        } else {
            tout[u] = timer;
            st.pop_back();
        }
    }

    vector<long long> acc(n + 1, 0);
    vector<int> touched;
    touched.reserve(n);
    while (q--) {
        int u, l, r;
        cin >> u >> l >> r;
        touched.clear();
        for (int t = tin[u]; t <= tout[u]; t++) {
            int v = flat[t];
            int c = col[v];
            if (l <= c && c <= r) {
                if (acc[c] == 0) touched.push_back(c);
                acc[c] = (acc[c] + w[v]) % MOD;
            }
        }
        long long res = 0;
        for (int c : touched) {
            res = (res + acc[c] * acc[c]) % MOD;
            acc[c] = 0;
        }
        cout << res << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_17())
