"""Batch 16 (synth_76 .. synth_80): Hybrid OI Boss Batch — Domino Cascade DP+SegTree, Manhattan L1 MST, Offline Tree Color-Replaced Distances, Lazy Affine Segment Tree, Segment-Tree 2-SAT Max-Min Spacing."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def _rand_tree_weighted_colored(n: int, r: random.Random, kind: str = "random") -> list[tuple[int, int, int, int]]:
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    es = []
    cmax = max(1, n - 1)
    for i in range(1, n):
        if kind == "chain":
            p = perm[i - 1]
        elif kind == "caterpillar":
            p = perm[max(0, i - r.randint(1, 3))]
        else:
            p = perm[r.randint(0, i - 1)]
        col = r.randint(1, min(cmax, 25 if r.random() < 0.5 else cmax))
        w = r.randint(1, 10**7)
        es.append((p, perm[i], col, w))
    return es


def make_batch_16() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_76: DOMINO CASCADE (1900) - Right-to-Left DP + Range-Max SegTree
    # ------------------------------------------------------------------
    def gen_76(n: int, seed: int, full_cascade: bool = False) -> str:
        r = random.Random(seed)
        xs = []
        cur = 0
        for _ in range(n):
            cur += r.randint(1, 5 if full_cascade else 50)
            xs.append(cur)
        hs = [r.randint(5, 15) if full_cascade else r.randint(1, 120) for _ in range(n)]
        lines = [str(n)] + [f"{xs[i]} {hs[i]}" for i in range(n)]
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_76",
        title="Domino Cascade",
        rating=1900,
        tags=["dp", "segment tree", "binary search", "data structures"],
        notes="Process dominoes from right to left: binary search rightmost directly hit index reach, then far[i] = max(reach, seg_max(i..reach)); answer for i is far[i] - i + 1.",
        story_and_spec="""N dominoes stand along a number line at strictly increasing integer coordinates X_1 < X_2 < ... < X_N. Domino i has positive height H_i.

If you tip domino i to the right, it falls and knocks over every domino j > i satisfying X_j <= X_i + H_i. Every domino that is knocked over also falls to the right and knocks over further dominoes by the same rule, creating a chain reaction.

For EACH starting domino i = 1, 2, ..., N (considered independently, starting with all dominoes upright), compute how many dominoes in total end up knocked over (including domino i itself).

Input
The first line contains N.
Each of the next N lines contains X_i and H_i (with X_1 < X_2 < ... < X_N).

Output
Print N space-separated integers: the total number of toppled dominoes when pushing domino 1, 2, ..., N.

Constraints
1 <= N <= 200000
0 <= X_1 < X_2 < ... < X_N <= 10^9
1 <= H_i <= 10^9""",
        samples=[
            "5\n1 2\n3 1\n5 4\n7 1\n10 5\n",
        ],
        small_inputs=[
            "1\n0 10\n",
            gen_76(300, 7601),
            gen_76(1800, 7602, full_cascade=True),
        ],
        large_inputs=[
            gen_76(200000, 7611, full_cascade=True),
            gen_76(200000, 7612, full_cascade=False),
            gen_76(200000, 7613, full_cascade=False),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> x(n), h(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> h[i];

    int S = 1;
    while (S < n) S <<= 1;
    vector<int> seg(2 * S, 0);
    auto seg_set = [&](int idx, int val) {
        int u = S + idx;
        seg[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = max(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_q = [&](int l, int r) {
        int res = l;
        for (l += S, r += S + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = max(res, seg[l++]);
            if (r & 1) res = max(res, seg[--r]);
        }
        return res;
    };

    vector<int> ans(n);
    for (int i = n - 1; i >= 0; i--) {
        int reach = (int)(upper_bound(x.begin(), x.end(), x[i] + h[i]) - x.begin()) - 1;
        int far = max(reach, seg_q(i, reach));
        seg_set(i, far);
        ans[i] = far - i + 1;
    }
    for (int i = 0; i < n; i++) cout << ans[i] << (i + 1 == n ? '\n' : ' ');
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
    vector<long long> x(n), h(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> h[i];
    for (int i = 0; i < n; i++) {
        long long lim = x[i] + h[i];
        int j = i;
        while (j + 1 < n && x[j + 1] <= lim) {
            j++;
            lim = max(lim, x[j] + h[j]);
        }
        cout << (j - i + 1) << (i + 1 == n ? '\n' : ' ');
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_77: OASIS PIPELINES (2300) - 2D L1 Manhattan Minimum Spanning Tree
    # ------------------------------------------------------------------
    def gen_77(n: int, cmax: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [str(n)] + [f"{r.randint(0, cmax)} {r.randint(0, cmax)}" for _ in range(n)]
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_77",
        title="Oasis Pipelines",
        rating=2300,
        tags=["geometry", "mst", "sweepline", "fenwick tree", "dsu"],
        notes="8-octant (D_4) sweepline + Fenwick min-query emits <= 8N candidate edges containing a valid Manhattan MST, solved by Kruskal in O(N log N).",
        story_and_spec="""In a desert basin there are N oases at integer coordinates (X_i, Y_i) for i = 1, 2, ..., N. Connecting oasis i and oasis j with an east-west / north-south pipeline costs their rectilinear (Manhattan) distance:
|X_i - X_j| + |Y_i - Y_j|.

Find the minimum total pipeline length required to connect all N oases into one connected network.

Input
The first line contains N.
Each of the next N lines contains X_i and Y_i.

Output
Print a single integer: the minimum total Manhattan spanning tree length.

Constraints
1 <= N <= 100000
0 <= X_i, Y_i <= 10^9""",
        samples=[
            "4\n0 0\n3 1\n1 4\n4 4\n",
            "3\n5 5\n5 5\n2 9\n",
        ],
        small_inputs=[
            "1\n42 99\n",
            gen_77(250, 50, 7701),
            gen_77(1500, 10**9, 7702),
        ],
        large_inputs=[
            gen_77(100000, 10**9, 7711),
            gen_77(100000, 10000, 7712),
            gen_77(100000, 10**9, 7713),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n), sz(n, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a; sz[a] += sz[b];
        return true;
    }
};

struct Edge {
    long long w;
    int u, v;
    bool operator<(const Edge& o) const { return w < o.w; }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> ox(n), oy(n);
    for (int i = 0; i < n; i++) cin >> ox[i] >> oy[i];
    if (n <= 1) {
        cout << 0 << '\n';
        return 0;
    }

    const long long INF = (1LL << 61);
    vector<Edge> edges;
    edges.reserve(8 * n);

    vector<long long> tx(n), ty(n), diffs(n);
    vector<int> ord(n);
    vector<pair<long long, int>> bit;

    for (int fx : {1, -1}) {
        for (int fy : {1, -1}) {
            for (int sw : {0, 1}) {
                diffs.resize(n);
                for (int i = 0; i < n; i++) {
                    long long x = fx * ox[i], y = fy * oy[i];
                    if (sw) swap(x, y);
                    tx[i] = x;
                    ty[i] = y;
                    diffs[i] = x - y;
                }
                sort(diffs.begin(), diffs.end());
                diffs.erase(unique(diffs.begin(), diffs.end()), diffs.end());
                int m = (int)diffs.size();

                iota(ord.begin(), ord.end(), 0);
                sort(ord.begin(), ord.end(), [&](int a, int b) {
                    if (ty[a] != ty[b]) return ty[a] > ty[b];
                    if (tx[a] != tx[b]) return tx[a] > tx[b];
                    return a > b;
                });

                bit.assign(m + 1, {INF, -1});
                auto upd = [&](int idx, pair<long long, int> val) {
                    for (; idx <= m; idx += idx & -idx)
                        if (val < bit[idx]) bit[idx] = val;
                };
                auto qry = [&](int idx) {
                    pair<long long, int> res = {INF, -1};
                    for (; idx > 0; idx -= idx & -idx)
                        if (bit[idx] < res) res = bit[idx];
                    return res;
                };

                for (int u : ord) {
                    int rk = (int)(lower_bound(diffs.begin(), diffs.end(), tx[u] - ty[u]) - diffs.begin());
                    int pos = m - rk; // reversed so prefix [1..pos] means tx - ty >= tx[u] - ty[u]
                    auto [_, v] = qry(pos);
                    if (v != -1) {
                        long long w = llabs(ox[u] - ox[v]) + llabs(oy[u] - oy[v]);
                        edges.push_back({w, u, v});
                    }
                    upd(pos, {tx[u] + ty[u], u});
                }
            }
        }
    }

    sort(edges.begin(), edges.end());
    DSU dsu(n);
    long long total = 0;
    int taken = 0;
    for (const auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            total += e.w;
            if (++taken == n - 1) break;
        }
    }
    cout << total << '\n';
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
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    const long long INF = (1LL << 61);
    vector<long long> best(n, INF);
    vector<char> used(n, 0);
    best[0] = 0;
    long long ans = 0;
    for (int it = 0; it < n; it++) {
        int u = -1;
        for (int i = 0; i < n; i++) {
            if (!used[i] && (u == -1 || best[i] < best[u])) u = i;
        }
        used[u] = 1;
        ans += best[u];
        for (int v = 0; v < n; v++) if (!used[v]) {
            long long d = llabs(x[u] - x[v]) + llabs(y[u] - y[v]);
            if (d < best[v]) best[v] = d;
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_78: TINTED HIGHWAYS (2100) - Offline Euler DFS Color-Replaced Tree Distance
    # ------------------------------------------------------------------
    def gen_78(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree_weighted_colored(n, r, kind)
        cmax = max(1, n - 1)
        lines = [f"{n} {q}"] + [f"{u} {v} {c} {w}" for u, v, c, w in es]
        for _ in range(q):
            a = r.randint(1, n)
            b = r.randint(1, n)
            col = r.randint(1, min(cmax, 25 if r.random() < 0.6 else cmax))
            nw = r.randint(0, 10**7)
            lines.append(f"{a} {b} {col} {nw}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_78",
        title="Tinted Highways",
        rating=2100,
        tags=["trees", "lca", "dfs and similar", "offline queries"],
        notes="ABC133F Colorful Tree: compute LCA w = lca(A, B), decompose query into +f(A) + f(B) - 2*f(w), and answer all queries in one root-to-leaf DFS maintaining (cnt[color], sum[color]) on the ancestor stack.",
        story_and_spec="""A kingdom has N cities numbered 1 to N connected by N - 1 bidirectional roads forming a tree. Road i connects U_i and V_i, is painted color C_i (1 <= C_i <= N - 1), and has base length W_i.

Answer Q independent hypothetical queries. Each query gives four integers A, B, K, and X: if EVERY road of color K had its length temporarily changed to X (while roads of all other colors keep their base lengths), what would be the total length of the unique simple path between city A and city B?

Input
The first line contains N and Q.
Each of the next N - 1 lines contains four integers U_i, V_i, C_i, and W_i.
Each of the next Q lines contains four integers A, B, K, and X.

Output
For each query print the hypothetical path length between A and B on its own line.

Constraints
2 <= N <= 180000
1 <= Q <= 180000
1 <= U_i, V_i, A, B <= N
1 <= C_i, K <= N - 1
0 <= W_i, X <= 10^7""",
        samples=[
            "5 3\n1 2 1 10\n2 3 2 20\n2 4 1 30\n4 5 3 40\n1 5 1 5\n3 5 2 100\n1 4 3 999\n",
        ],
        small_inputs=[
            "2 2\n1 2 1 50\n1 2 1 7\n1 1 1 7\n",
            gen_78(300, 400, 7801),
            gen_78(1400, 1400, 7802, "chain"),
        ],
        large_inputs=[
            gen_78(180000, 180000, 7811, "chain"),
            gen_78(180000, 180000, 7812, "caterpillar"),
            gen_78(180000, 180000, 7813, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct AdjEdge { int to, col; long long w; };
struct Req { int qid, col, coeff; long long nw; };

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<AdjEdge>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, c; long long w;
        cin >> u >> v >> c >> w;
        adj[u].push_back({v, c, w});
        adj[v].push_back({u, c, w});
    }

    // Iterative Euler tour for O(1) RMQ LCA + base root distances
    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
    vector<long long> rdist(n + 1, 0);
    euler.reserve(2 * n);
    {
        vector<pair<int, int>> st = {{1, 0}};
        while (!st.empty()) {
            auto& [u, idx] = st.back();
            if (idx == 0) {
                first_pos[u] = (int)euler.size();
                euler.push_back(u);
            }
            if (idx < (int)adj[u].size()) {
                auto e = adj[u][idx++];
                if (e.to == par[u]) continue;
                par[e.to] = u;
                dep[e.to] = dep[u] + 1;
                rdist[e.to] = rdist[u] + e.w;
                st.push_back({e.to, 0});
            } else {
                st.pop_back();
                if (!st.empty()) euler.push_back(st.back().first);
            }
        }
    }
    int m = (int)euler.size(), LOG = __lg(m) + 1;
    vector<vector<int>> rmq(LOG, vector<int>(m));
    rmq[0] = euler;
    for (int k = 1; k < LOG; k++) {
        int h = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= m; i++) {
            int a = rmq[k - 1][i], b = rmq[k - 1][i + h];
            rmq[k][i] = dep[a] < dep[b] ? a : b;
        }
    }
    auto lca = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        return dep[a] < dep[b] ? a : b;
    };

    vector<vector<Req>> reqs(n + 1);
    for (int i = 0; i < q; i++) {
        int a, b, col; long long nw;
        cin >> a >> b >> col >> nw;
        int w = lca(a, b);
        reqs[a].push_back({i, col, +1, nw});
        reqs[b].push_back({i, col, +1, nw});
        reqs[w].push_back({i, col, -2, nw});
    }

    vector<long long> ans(q, 0), sum_col(n + 1, 0);
    vector<int> cnt_col(n + 1, 0);

    struct Frame { int u, idx, in_col; long long in_w; };
    vector<Frame> st = {{1, 0, 0, 0}};
    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.idx == 0) {
            if (f.in_col) {
                cnt_col[f.in_col]++;
                sum_col[f.in_col] += f.in_w;
            }
            for (const auto& rq : reqs[u]) {
                long long cur = rdist[u] - sum_col[rq.col] + rq.nw * cnt_col[rq.col];
                ans[rq.qid] += rq.coeff * cur;
            }
        }
        bool pushed = false;
        while (f.idx < (int)adj[u].size()) {
            auto e = adj[u][f.idx++];
            if (e.to == par[u]) continue;
            st.push_back({e.to, 0, e.col, e.w});
            pushed = true;
            break;
        }
        if (!pushed) {
            if (f.in_col) {
                cnt_col[f.in_col]--;
                sum_col[f.in_col] -= f.in_w;
            }
            st.pop_back();
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
    struct E { int to, col; long long w; };
    vector<vector<E>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, c; long long w;
        cin >> u >> v >> c >> w;
        adj[u].push_back({v, c, w});
        adj[v].push_back({u, c, w});
    }
    vector<int> par(n + 1, 0), dep(n + 1, 0), pcol(n + 1, 0), ord = {1};
    vector<long long> pw(n + 1, 0);
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto e : adj[u]) if (e.to != par[u]) {
            par[e.to] = u;
            dep[e.to] = dep[u] + 1;
            pcol[e.to] = e.col;
            pw[e.to] = e.w;
            ord.push_back(e.to);
        }
    }
    while (q--) {
        int a, b, k; long long nw;
        cin >> a >> b >> k >> nw;
        long long tot = 0;
        auto step_up = [&](int& x) {
            tot += (pcol[x] == k ? nw : pw[x]);
            x = par[x];
        };
        while (dep[a] > dep[b]) step_up(a);
        while (dep[b] > dep[a]) step_up(b);
        while (a != b) { step_up(a); step_up(b); }
        cout << tot << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_79: AFFINE LEDGER (1900) - Lazy Affine Segment Tree
    # ------------------------------------------------------------------
    def gen_79(n: int, q: int, seed: int, wide: bool = False) -> str:
        r = random.Random(seed)
        MOD = 998244353
        a = [r.randrange(MOD) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, a))]
        for i in range(q):
            if wide and r.random() < 0.85:
                l = r.randint(1, max(1, n // 6))
                rr = r.randint(max(l, 5 * n // 6), n)
            else:
                l, rr = r.randint(1, n), r.randint(1, n)
                if l > rr:
                    l, rr = rr, l
            if i < q - 1 and r.random() < 0.55:
                b = 0 if r.random() < 0.05 else r.randrange(MOD)
                c = r.randrange(MOD)
                lines.append(f"1 {l} {rr} {b} {c}")
            else:
                lines.append(f"2 {l} {rr}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_79",
        title="Affine Ledger",
        rating=1900,
        tags=["data structures", "segment tree", "lazy propagation"],
        notes="Lazy segment tree composing affine maps f2(f1(x)) = (b2*b1) * x + (b2*c1 + c2) mod 998244353 with range sum.",
        story_and_spec="""Maintain an array A_1, A_2, ..., A_N of integers modulo 998244353 under Q operations of two types:
- 1 L R B C: for every index i from L to R inclusive, simultaneously update A_i <- (B * A_i + C) mod 998244353.
- 2 L R: compute (A_L + A_{L+1} + ... + A_R) mod 998244353.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines describes an operation in one of the two formats above. At least one operation is of type 2.

Output
For each operation of type 2, print the range sum modulo 998244353 on its own line.

Constraints
1 <= N, Q <= 200000
0 <= A_i, B, C < 998244353
1 <= L <= R <= N""",
        samples=[
            "5 5\n1 2 3 4 5\n2 1 5\n1 2 4 2 3\n2 1 5\n1 1 3 0 7\n2 2 5\n",
        ],
        small_inputs=[
            "1 3\n5\n1 1 1 3 4\n2 1 1\n1 1 1 0 0\n",
            gen_79(300, 400, 7901),
            gen_79(1500, 1500, 7902, wide=True),
        ],
        large_inputs=[
            gen_79(200000, 200000, 7911, wide=True),
            gen_79(200000, 200000, 7912, wide=False),
            gen_79(200000, 200000, 7913, wide=True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<long long> sum(4 * n + 5, 0), mul(4 * n + 5, 1), add(4 * n + 5, 0);

    auto apply = [&](int u, int len, long long b, long long c) {
        sum[u] = (sum[u] * b + c * len) % MOD;
        mul[u] = (mul[u] * b) % MOD;
        add[u] = (add[u] * b + c) % MOD;
    };
    auto push = [&](int u, int l, int r) {
        if (mul[u] != 1 || add[u] != 0) {
            int mid = (l + r) >> 1;
            apply(u << 1, mid - l + 1, mul[u], add[u]);
            apply(u << 1 | 1, r - mid, mul[u], add[u]);
            mul[u] = 1;
            add[u] = 0;
        }
    };

    function<void(int, int, int)> build = [&](int u, int l, int r) {
        if (l == r) { sum[u] = a[l] % MOD; return; }
        int mid = (l + r) >> 1;
        build(u << 1, l, mid);
        build(u << 1 | 1, mid + 1, r);
        sum[u] = (sum[u << 1] + sum[u << 1 | 1]) % MOD;
    };
    function<void(int, int, int, int, int, long long, long long)> upd =
        [&](int u, int l, int r, int ql, int qr, long long b, long long c) {
            if (ql <= l && r <= qr) { apply(u, r - l + 1, b, c); return; }
            push(u, l, r);
            int mid = (l + r) >> 1;
            if (ql <= mid) upd(u << 1, l, mid, ql, qr, b, c);
            if (qr > mid) upd(u << 1 | 1, mid + 1, r, ql, qr, b, c);
            sum[u] = (sum[u << 1] + sum[u << 1 | 1]) % MOD;
        };
    function<long long(int, int, int, int, int)> qry =
        [&](int u, int l, int r, int ql, int qr) -> long long {
            if (ql <= l && r <= qr) return sum[u];
            push(u, l, r);
            int mid = (l + r) >> 1;
            long long res = 0;
            if (ql <= mid) res += qry(u << 1, l, mid, ql, qr);
            if (qr > mid) res += qry(u << 1 | 1, mid + 1, r, ql, qr);
            return res % MOD;
        };

    build(1, 1, n);
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int l, r; long long b, c;
            cin >> l >> r >> b >> c;
            upd(1, 1, n, l, r, b, c);
        } else {
            int l, r;
            cin >> l >> r;
            cout << qry(1, 1, n, l, r) << '\n';
        }
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 998244353LL;

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
            int l, r; long long b, c;
            cin >> l >> r >> b >> c;
            for (int i = l; i <= r; i++) a[i] = (b * a[i] + c) % MOD;
        } else {
            int l, r;
            cin >> l >> r;
            long long s = 0;
            for (int i = l; i <= r; i++) s += a[i];
            cout << (s % MOD) << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_80: TOWER SPACING (2300) - Binary Search + Segment-Tree 2-SAT (ARC069F)
    # ------------------------------------------------------------------
    def gen_80(n: int, cmax: int, seed: int, planted: bool = False) -> str:
        r = random.Random(seed)
        lines = [str(n)]
        if planted:
            # Plant a valid choice with spacing ~ cmax // (2*n) and random distractors
            step = max(1, cmax // (2 * n))
            good = [i * step + r.randint(0, max(0, step // 3)) for i in range(n)]
            r.shuffle(good)
            for i in range(n):
                other = r.randint(0, cmax)
                if r.random() < 0.5:
                    lines.append(f"{good[i]} {other}")
                else:
                    lines.append(f"{other} {good[i]}")
        else:
            for _ in range(n):
                lines.append(f"{r.randint(0, cmax)} {r.randint(0, cmax)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_80",
        title="Tower Spacing",
        rating=2300,
        tags=["2-sat", "segment tree", "binary search", "graphs", "scc"],
        notes="ARC069F Flags: binary search minimum spacing D; sort all 2N candidate coordinates and use a virtual-clause Segment Tree + Tarjan SCC to add range conflict clauses in O(N log N log V).",
        story_and_spec="""Along a straight highway, a telecom company must erect N relay towers. For each i = 1, 2, ..., N, tower i has two permitted site coordinates, A_i and B_i, and you must build tower i at EITHER X_i = A_i OR X_i = B_i.

To minimize signal interference, you want the minimum distance between any two erected towers,
min_{1 <= i < j <= N} |X_i - X_j|,
to be as LARGE as possible.

Find the maximum achievable value of min_{1 <= i < j <= N} |X_i - X_j|.

Input
The first line contains N.
Each of the next N lines contains two integers A_i and B_i.

Output
Print a single integer: the maximum possible minimum pairwise distance between chosen tower sites.

Constraints
2 <= N <= 20000
0 <= A_i, B_i <= 10^9""",
        samples=[
            "3\n1 9\n3 6\n2 12\n",
            "2\n5 5\n5 5\n",
        ],
        small_inputs=[
            "2\n0 100\n50 50\n",
            gen_80(120, 5000, 8001, planted=True),
            gen_80(350, 10**9, 8002, planted=False),
        ],
        large_inputs=[
            gen_80(20000, 10**9, 8011, planted=True),
            gen_80(20000, 10**9, 8012, planted=False),
            gen_80(18000, 10**7, 8013, planted=True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    const int M = 2 * n;
    struct Slot {
        long long x;
        int lit;
        bool operator<(const Slot& o) const { return x < o.x; }
    };
    vector<Slot> slots(M);
    for (int i = 0; i < n; i++) {
        long long a, b;
        cin >> a >> b;
        slots[2 * i] = {a, 2 * i};
        slots[2 * i + 1] = {b, 2 * i + 1};
    }
    sort(slots.begin(), slots.end());
    vector<long long> xs(M);
    for (int i = 0; i < M; i++) xs[i] = slots[i].x;

    // Graph layout:
    // 0 .. M-1: real literals (2*i and 2*i+1)
    // M + (1 .. 2*M - 1): iterative segment-tree nodes where leaf M + (M + p) -> (slots[p].lit ^ 1)
    const int V = M + 2 * M;
    const int MAXE = 2 * M + M * 36 + 100;
    vector<int> head(V, -1), to(MAXE), nxt(MAXE);
    int ecnt = 0;
    auto add_edge = [&](int u, int v) {
        to[ecnt] = v;
        nxt[ecnt] = head[u];
        head[u] = ecnt++;
    };

    // Prebuild static segment-tree edges once; snapshot head_base and ecnt_base!
    for (int u = 1; u < M; u++) {
        add_edge(M + u, M + 2 * u);
        add_edge(M + u, M + 2 * u + 1);
    }
    for (int p = 0; p < M; p++) {
        add_edge(M + (M + p), slots[p].lit ^ 1);
    }
    vector<int> head_base = head;
    const int ecnt_base = ecnt;

    vector<int> dfn(V, 0), low(V, 0), comp(V, -1), stk;
    vector<char> in_stk(V, 0);
    stk.reserve(V);
    struct Frame { int u, e, child; bool ret; };
    vector<Frame> call;
    call.reserve(V);

    auto add_range = [&](int src, int l, int r) {
        for (l += M, r += M + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) add_edge(src, M + (l++));
            if (r & 1) add_edge(src, M + (--r));
        }
    };

    auto ok = [&](long long D) -> bool {
        if (D <= 0) return true;
        // Only real literals 0..M-1 got extra edges; restore them in O(M)!
        for (int i = 0; i < M; i++) head[i] = head_base[i];
        ecnt = ecnt_base;

        int L = 0, R = 0;
        for (int p = 0; p < M; p++) {
            while (xs[p] - xs[L] >= D) L++;
            while (R + 1 < M && xs[R + 1] - xs[p] < D) R++;
            int lit = slots[p].lit;
            if (L <= p - 1) add_range(lit, L, p - 1);
            if (p + 1 <= R) add_range(lit, p + 1, R);
        }

        fill(dfn.begin(), dfn.end(), 0);
        fill(comp.begin(), comp.end(), -1);
        int timer = 0, scc_id = 0;
        stk.clear();

        for (int s = 0; s < M; s++) {
            if (dfn[s]) continue;
            call.push_back({s, head[s], -1, false});
            while (!call.empty()) {
                auto& f = call.back();
                int u = f.u;
                if (!dfn[u]) {
                    dfn[u] = low[u] = ++timer;
                    stk.push_back(u);
                    in_stk[u] = 1;
                }
                if (f.ret) {
                    low[u] = min(low[u], low[f.child]);
                    f.ret = false;
                }
                bool pushed = false;
                while (f.e != -1) {
                    int v = to[f.e];
                    f.e = nxt[f.e];
                    if (!dfn[v]) {
                        f.child = v;
                        f.ret = true;
                        call.push_back({v, head[v], -1, false});
                        pushed = true;
                        break;
                    } else if (in_stk[v]) {
                        low[u] = min(low[u], dfn[v]);
                    }
                }
                if (!pushed) {
                    if (low[u] == dfn[u]) {
                        while (true) {
                            int x = stk.back(); stk.pop_back();
                            in_stk[x] = 0;
                            comp[x] = scc_id;
                            if (x == u) break;
                        }
                        scc_id++;
                    }
                    call.pop_back();
                }
            }
        }

        for (int i = 0; i < n; i++) {
            if (comp[2 * i] == comp[2 * i + 1]) return false;
        }
        return true;
    };

    long long lo = 0, hi = xs[M - 1] - xs[0];
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (ok(mid)) lo = mid;
        else hi = mid - 1;
    }
    cout << lo << '\n';
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
    const int M = 2 * n;
    vector<long long> pos(M);
    long long mn = (1LL << 60), mx = 0;
    for (int i = 0; i < n; i++) {
        cin >> pos[2 * i] >> pos[2 * i + 1];
        mn = min({mn, pos[2 * i], pos[2 * i + 1]});
        mx = max({mx, pos[2 * i], pos[2 * i + 1]});
    }

    // O(N^2) implicit-graph Tarjan 2-SAT using O(N) memory
    vector<int> dfn(M), low(M), comp(M), stk;
    vector<char> in_stk(M, 0);
    struct Frame { int u, v_idx, child; bool ret; };
    vector<Frame> call;

    auto ok = [&](long long D) -> bool {
        if (D <= 0) return true;
        fill(dfn.begin(), dfn.end(), 0);
        fill(comp.begin(), comp.end(), -1);
        int timer = 0, scc_id = 0;
        stk.clear();

        for (int s = 0; s < M; s++) {
            if (dfn[s]) continue;
            call.push_back({s, 0, -1, false});
            while (!call.empty()) {
                auto& f = call.back();
                int u = f.u;
                if (!dfn[u]) {
                    dfn[u] = low[u] = ++timer;
                    stk.push_back(u);
                    in_stk[u] = 1;
                }
                if (f.ret) {
                    low[u] = min(low[u], low[f.child]);
                    f.ret = false;
                }
                bool pushed = false;
                while (f.v_idx < M) {
                    int q = f.v_idx++;
                    if (q == u || llabs(pos[u] - pos[q]) >= D) continue;
                    int v = q ^ 1;
                    if (!dfn[v]) {
                        f.child = v;
                        f.ret = true;
                        call.push_back({v, 0, -1, false});
                        pushed = true;
                        break;
                    } else if (in_stk[v]) {
                        low[u] = min(low[u], dfn[v]);
                    }
                }
                if (!pushed) {
                    if (low[u] == dfn[u]) {
                        while (true) {
                            int x = stk.back(); stk.pop_back();
                            in_stk[x] = 0;
                            comp[x] = scc_id;
                            if (x == u) break;
                        }
                        scc_id++;
                    }
                    call.pop_back();
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (comp[2 * i] == comp[2 * i + 1]) return false;
        }
        return true;
    };

    long long lo = 0, hi = mx - mn;
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (ok(mid)) lo = mid;
        else hi = mid - 1;
    }
    cout << lo << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_16())
