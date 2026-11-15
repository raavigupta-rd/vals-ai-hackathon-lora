"""Batch 10 (synth_46 .. synth_50): Trees II — K-th Path Node, Dynamic Nearest Marked (Centroid Decomp), HLD Path Max, Forest Diameter Merging, Exact-K Paths."""

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


def make_batch_10() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_46: PILGRIM WAYSTATIONS (1800) - K-th Vertex on Tree Path
    # ------------------------------------------------------------------
    def gen_46(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        lines = [f"{n} {q}"] + [f"{u} {v}" for u, v in es]
        for _ in range(q):
            u, v = r.randint(1, n), r.randint(1, n)
            k = r.randint(0, n) if r.random() < 0.25 else r.randint(0, max(1, n // 3))
            lines.append(f"{u} {v} {k}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_46",
        title="Pilgrim Waystations",
        rating=1800,
        tags=["trees", "lca", "binary lifting"],
        notes="Compute LCA(u, v); if K <= d(u, w) jump K steps up from u, else jump d(u, w) + d(v, w) - K steps up from v.",
        story_and_spec="""A mountain sanctuary has N shrines numbered 1 to N connected by N - 1 bidirectional footpaths forming a tree.

You are asked Q route queries. Each query specifies start shrine U, destination shrine V, and step count K (K >= 0): walking along the unique simple path from U to V one footpath per step, report which shrine is reached after K steps (for K = 0 report U), or report -1 if the simple path from U to V has fewer than K edges.

Input
The first line contains N and Q.
Each of the next N - 1 lines contains two shrines connected by a footpath.
Each of the next Q lines contains U, V, and K.

Output
For each query print the shrine reached after K steps (or -1) on its own line.

Constraints
1 <= N, Q <= 200000
1 <= U, V <= N
0 <= K <= N""",
        samples=[
            "6 5\n1 2\n2 3\n2 4\n4 5\n1 6\n5 6 0\n5 6 2\n5 6 4\n5 6 5\n3 5 2\n",
        ],
        expected_samples=["5\n2\n6\n-1\n4\n"],
        small_inputs=[
            "1 2\n1 1 0\n1 1 1\n",
            gen_46(400, 500, 4601),
            gen_46(1500, 1500, 4602, "chain"),
        ],
        large_inputs=[
            gen_46(200000, 200000, 4611, "chain"),
            gen_46(200000, 200000, 4612, "caterpillar"),
            gen_46(200000, 200000, 4613, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int LOG = 1;
    while ((1 << LOG) <= n) LOG++;
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<int> dep(n + 1, 0), qv = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = qv[i];
        for (int v : adj[u]) if (!dep[v]) {
            dep[v] = dep[u] + 1;
            up[0][v] = u;
            qv.push_back(v);
        }
    }
    for (int b = 1; b < LOG; b++)
        for (int i = 1; i <= n; i++)
            up[b][i] = up[b - 1][up[b - 1][i]];

    auto kth_anc = [&](int u, int k) {
        for (int b = 0; b < LOG; b++) if ((k >> b) & 1) u = up[b][u];
        return u;
    };
    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        u = kth_anc(u, dep[u] - dep[v]);
        if (u == v) return u;
        for (int b = LOG - 1; b >= 0; b--)
            if (up[b][u] != up[b][v]) { u = up[b][u]; v = up[b][v]; }
        return up[0][u];
    };

    while (q--) {
        int u, v, k;
        cin >> u >> v >> k;
        int w = lca(u, v);
        int du = dep[u] - dep[w], dv = dep[v] - dep[w];
        if (k > du + dv) cout << -1 << '\n';
        else if (k <= du) cout << kth_anc(u, k) << '\n';
        else cout << kth_anc(v, du + dv - k) << '\n';
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
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1, 0), dep(n + 1, 0), qv = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = qv[i];
        for (int v : adj[u]) if (!dep[v]) {
            dep[v] = dep[u] + 1;
            par[v] = u;
            qv.push_back(v);
        }
    }
    while (q--) {
        int u, v, k;
        cin >> u >> v >> k;
        vector<int> L, R;
        int a = u, b = v;
        while (dep[a] > dep[b]) { L.push_back(a); a = par[a]; }
        while (dep[b] > dep[a]) { R.push_back(b); b = par[b]; }
        while (a != b) { L.push_back(a); R.push_back(b); a = par[a]; b = par[b]; }
        L.push_back(a);
        L.insert(L.end(), R.rbegin(), R.rend());
        cout << (k < (int)L.size() ? L[k] : -1) << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_47: PLAGUE INFIRMARIES (2200) - Centroid Decomposition (Xenia & Tree)
    # ------------------------------------------------------------------
    def gen_47(n: int, q: int, seed: int, kind: str = "random", adv: bool = False) -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        chain_nodes = [es[0][0]] + [v for _, v in es] if es else [1]
        lines = [f"{n} {q}"] + [f"{u} {v}" for u, v in es]
        for i in range(q):
            if i < q - 1 and r.random() < 0.25:
                u = chain_nodes[r.randint(0, min(30, n - 1))] if adv else r.randint(1, n)
                lines.append(f"1 {u}")
            else:
                v = chain_nodes[r.randint(n // 2, n - 1)] if adv else r.randint(1, n)
                lines.append(f"2 {v}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_47",
        title="Plague Infirmaries",
        rating=2200,
        tags=["trees", "centroid decomposition", "lca", "divide and conquer"],
        notes="Centroid decomposition storing best[c] = min distance from centroid c to any active infirmary in its centroid subtree; O(log N) per update and query.",
        story_and_spec="""A kingdom has N towns numbered 1 to N connected by N - 1 bidirectional roads forming a tree. Initially, ONLY town 1 has an open infirmary.

Process Q events of two types:
- 1 U: open an infirmary in town U (if one is already open in U, it stays open).
- 2 V: report the minimum number of roads on the path from town V to ANY currently open infirmary.

Input
The first line contains N and Q.
Each of the next N - 1 lines contains two towns connected by a road.
Each of the next Q lines describes an event in one of the two formats above. At least one event is of type 2.

Output
For each event of type 2, print the shortest distance to an open infirmary on its own line.

Constraints
1 <= N, Q <= 160000
1 <= U, V <= N""",
        samples=[
            "5 4\n1 2\n2 3\n2 4\n4 5\n2 5\n1 4\n2 5\n2 3\n",
        ],
        expected_samples=["3\n1\n2\n"],
        small_inputs=[
            "2 3\n1 2\n2 2\n1 2\n2 2\n",
            gen_47(400, 500, 4701),
            gen_47(1400, 1500, 4702, "chain"),
        ],
        large_inputs=[
            gen_47(150000, 150000, 4711, "chain", adv=True),
            gen_47(160000, 160000, 4712, "caterpillar"),
            gen_47(160000, 160000, 4713, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Euler tour RMQ LCA for O(1) tree distance
    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
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
                int v = adj[u][idx++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] + 1;
                st.push_back({v, 0});
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
    auto dist = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        int w = dep[a] < dep[b] ? a : b;
        return dep[u] + dep[v] - 2 * dep[w];
    };

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), cpar(n + 1, 0), tmp_ord, tmp_par(n + 1, 0);

    function<void(int, int)> build_cd = [&](int start, int p_cent) {
        tmp_ord.clear();
        tmp_ord.push_back(start);
        tmp_par[start] = 0;
        for (size_t i = 0; i < tmp_ord.size(); i++) {
            int u = tmp_ord[i];
            sz[u] = 1;
            for (int v : adj[u]) if (!dead[v] && v != tmp_par[u]) {
                tmp_par[v] = u;
                tmp_ord.push_back(v);
            }
        }
        int tot = (int)tmp_ord.size();
        for (int i = tot - 1; i >= 1; i--) sz[tmp_par[tmp_ord[i]]] += sz[tmp_ord[i]];
        int c = start;
        while (true) {
            int nxt = 0;
            for (int v : adj[c]) if (!dead[v] && v != tmp_par[c] && sz[v] > tot / 2) {
                nxt = v; break;
            }
            if (!nxt) break;
            c = nxt;
        }
        dead[c] = 1;
        cpar[c] = p_cent;
        for (int v : adj[c]) if (!dead[v]) build_cd(v, c);
    };
    build_cd(1, 0);

    const int INF = 1e9;
    vector<int> best(n + 1, INF);
    auto activate = [&](int u) {
        for (int c = u; c != 0; c = cpar[c]) best[c] = min(best[c], dist(u, c));
    };
    auto query = [&](int u) {
        int ans = INF;
        for (int c = u; c != 0; c = cpar[c]) ans = min(ans, best[c] + dist(u, c));
        return ans;
    };

    activate(1);
    while (q--) {
        int t, u;
        cin >> t >> u;
        if (t == 1) activate(u);
        else cout << query(u) << '\n';
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
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<char> open(n + 1, 0);
    open[1] = 1;
    vector<int> d(n + 1, -1), qv;
    while (q--) {
        int t, u; cin >> t >> u;
        if (t == 1) open[u] = 1;
        else {
            if (open[u]) { cout << 0 << '\n'; continue; }
            qv.assign(1, u);
            d[u] = 0;
            int ans = -1;
            for (size_t i = 0; i < qv.size(); i++) {
                int x = qv[i];
                for (int y : adj[x]) if (d[y] == -1) {
                    d[y] = d[x] + 1;
                    qv.push_back(y);
                    if (open[y]) { ans = d[y]; break; }
                }
                if (ans != -1) break;
            }
            for (int x : qv) d[x] = -1;
            cout << ans << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_48: ROYAL COURIER PATHS (2000) - Heavy-Light Decomposition (Path Max)
    # ------------------------------------------------------------------
    def gen_48(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        a = [r.randint(1, 10**9) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, a))] + [f"{u} {v}" for u, v in es]
        for i in range(q):
            if i < q - 1 and r.random() < 0.35:
                lines.append(f"1 {r.randint(1, n)} {r.randint(1, 10**9)}")
            else:
                lines.append(f"2 {r.randint(1, n)} {r.randint(1, n)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_48",
        title="Royal Courier Paths",
        rating=2000,
        tags=["trees", "heavy-light decomposition", "segment tree"],
        notes="Heavy-Light Decomposition + iterative max segment tree for vertex updates and path maximum queries.",
        story_and_spec="""A highland kingdom has N forts numbered 1 to N connected by N - 1 bidirectional roads forming a tree. Initially fort i charges courier toll A_i.

Process Q operations of two types:
- 1 U X: update the toll charged at fort U to X.
- 2 U V: report the MAXIMUM toll among all forts along the unique simple path between U and V (inclusive).

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next N - 1 lines contains two forts connected by a road.
Each of the next Q lines describes an operation in one of the two formats above. At least one operation is of type 2.

Output
For each operation of type 2, print the maximum path toll on its own line.

Constraints
1 <= N, Q <= 200000
1 <= A_i, X <= 10^9
1 <= U, V <= N""",
        samples=[
            "5 4\n10 50 20 40 30\n1 2\n1 3\n2 4\n2 5\n2 4 3\n1 2 15\n2 4 3\n2 4 5\n",
        ],
        expected_samples=["50\n40\n40\n"],
        small_inputs=[
            "1 2\n9\n2 1 1\n1 1 5\n",
            gen_48(400, 500, 4801),
            gen_48(1500, 1500, 4802, "chain"),
        ],
        large_inputs=[
            gen_48(200000, 200000, 4811, "chain"),
            gen_48(200000, 200000, 4812, "caterpillar"),
            gen_48(200000, 200000, 4813, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), dep(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0), ord = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] + 1;
            ord.push_back(v);
        }
    }
    for (int i = n - 1; i >= 1; i--) {
        int u = ord[i];
        sz[par[u]] += sz[u];
        if (sz[u] > sz[heavy[par[u]]]) heavy[par[u]] = u;
    }

    vector<int> head(n + 1, 0), pos(n + 1, 0);
    int cur_pos = 0;
    {
        vector<pair<int, int>> st = {{1, 1}};
        while (!st.empty()) {
            auto [u, h] = st.back(); st.pop_back();
            for (int x = u; x != 0; x = heavy[x]) {
                head[x] = h;
                pos[x] = cur_pos++;
                for (int v : adj[x]) if (v != par[x] && v != heavy[x]) st.push_back({v, v});
            }
        }
    }

    int S = 1;
    while (S < n) S <<= 1;
    vector<int> seg(2 * S, 0);
    for (int i = 1; i <= n; i++) seg[S + pos[i]] = a[i];
    for (int i = S - 1; i >= 1; i--) seg[i] = max(seg[2 * i], seg[2 * i + 1]);

    auto seg_set = [&](int idx, int val) {
        int u = S + idx;
        seg[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = max(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_q = [&](int l, int r) {
        int res = 0;
        for (l += S, r += S + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = max(res, seg[l++]);
            if (r & 1) res = max(res, seg[--r]);
        }
        return res;
    };

    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int u, x; cin >> u >> x;
            seg_set(pos[u], x);
        } else {
            int u, v; cin >> u >> v;
            int ans = 0;
            while (head[u] != head[v]) {
                if (dep[head[u]] < dep[head[v]]) swap(u, v);
                ans = max(ans, seg_q(pos[head[u]], pos[u]));
                u = par[head[u]];
            }
            if (dep[u] > dep[v]) swap(u, v);
            ans = max(ans, seg_q(pos[u], pos[v]));
            cout << ans << '\n';
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
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1, 0), dep(n + 1, 0), ord = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] + 1;
            ord.push_back(v);
        }
    }
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int u, x; cin >> u >> x;
            a[u] = x;
        } else {
            int u, v; cin >> u >> v;
            int mx = 0;
            while (dep[u] > dep[v]) { mx = max(mx, a[u]); u = par[u]; }
            while (dep[v] > dep[u]) { mx = max(mx, a[v]); v = par[v]; }
            while (u != v) { mx = max({mx, a[u], a[v]}); u = par[u]; v = par[v]; }
            mx = max(mx, a[u]);
            cout << mx << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_49: FOREST EXPANSION (2100) - Offline LCA + DSU Diameter Endpoints
    # ------------------------------------------------------------------
    def gen_49(n: int, seed: int, kind: str = "random", shuffle_edges: bool = True) -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        if shuffle_edges:
            r.shuffle(es)
        return f"{n}\n" + "\n".join(f"{u} {v}" for u, v in es) + "\n"

    specs.append(ProblemSpec(
        pid="synth_49",
        title="Forest Expansion",
        rating=2100,
        tags=["trees", "dsu", "lca", "diameter"],
        notes="Prebuild the final tree for O(1) RMQ LCA distances, then merge components in DSU tracking the two diameter endpoints {e1, e2} of each component (only 6 candidate pairs).",
        story_and_spec="""Initially there are N isolated outposts numbered 1 to N and no roads. One by one, N - 1 bidirectional roads are constructed; it is guaranteed that after all N - 1 roads are built, the outposts form a single tree.

Right after the i-th road (connecting U_i and V_i) is built, report the DIAMETER (the maximum number of roads on a simple path between any two outposts) of the connected component that now contains U_i and V_i.

Input
The first line contains N.
Each of the next N - 1 lines contains U_i and V_i.

Output
Print N - 1 lines: after each road is added, print the diameter of the newly merged component.

Constraints
2 <= N <= 200000
1 <= U_i, V_i <= N""",
        samples=[
            "5\n1 2\n3 4\n2 3\n2 5\n",
        ],
        expected_samples=["1\n1\n3\n3\n"],
        small_inputs=[
            "2\n1 2\n",
            gen_49(400, 4901),
            gen_49(1400, 4902, "chain"),
        ],
        large_inputs=[
            gen_49(180000, 4911, "chain", shuffle_edges=False),
            gen_49(200000, 4912, "caterpillar"),
            gen_49(200000, 4913, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<pair<int, int>> es(n - 1);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        cin >> es[i].first >> es[i].second;
        adj[es[i].first].push_back(es[i].second);
        adj[es[i].second].push_back(es[i].first);
    }

    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
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
                int v = adj[u][idx++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] + 1;
                st.push_back({v, 0});
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
    auto dist = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        int w = dep[a] < dep[b] ? a : b;
        return dep[u] + dep[v] - 2 * dep[w];
    };

    vector<int> p(n + 1), sz(n + 1, 1), e1(n + 1), e2(n + 1), diam(n + 1, 0);
    iota(p.begin(), p.end(), 0);
    iota(e1.begin(), e1.end(), 0);
    iota(e2.begin(), e2.end(), 0);
    function<int(int)> findp = [&](int x) { return p[x] == x ? x : p[x] = findp(p[x]); };

    for (auto [u, v] : es) {
        int a = findp(u), b = findp(v);
        if (sz[a] < sz[b]) swap(a, b);
        int cand[4] = {e1[a], e2[a], e1[b], e2[b]};
        int best_d = -1, ba = cand[0], bb = cand[0];
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                int d = dist(cand[i], cand[j]);
                if (d > best_d) { best_d = d; ba = cand[i]; bb = cand[j]; }
            }
        }
        p[b] = a;
        sz[a] += sz[b];
        e1[a] = ba; e2[a] = bb; diam[a] = best_d;
        cout << best_d << '\n';
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
    vector<vector<int>> adj(n + 1);
    vector<int> d(n + 1, -1), qv;
    auto bfs = [&](int s) -> pair<int, int> {
        qv.assign(1, s);
        d[s] = 0;
        int far = s;
        for (size_t i = 0; i < qv.size(); i++) {
            int u = qv[i];
            if (d[u] > d[far]) far = u;
            for (int v : adj[u]) if (d[v] == -1) {
                d[v] = d[u] + 1;
                qv.push_back(v);
            }
        }
        int mx = d[far];
        for (int x : qv) d[x] = -1;
        return {far, mx};
    };
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
        auto [f, _] = bfs(u);
        auto [__, dm] = bfs(f);
        cout << dm << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_50: EXACT-LENGTH TRAILS (2100) - Centroid Decomp / HLD Depth Merge
    # ------------------------------------------------------------------
    def gen_50(n: int, k: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        return f"{n} {k}\n" + "\n".join(f"{u} {v}" for u, v in es) + "\n"

    specs.append(ProblemSpec(
        pid="synth_50",
        title="Exact-Length Trails",
        rating=2100,
        tags=["trees", "centroid decomposition", "divide and conquer", "dsu on tree"],
        notes="Centroid decomposition counting pairs of nodes at depths d and K - d across subtrees in O(N log N).",
        story_and_spec="""A national forest has N clearings numbered 1 to N connected by N - 1 bidirectional trails forming a tree. Count how many unordered pairs of distinct clearings {u, v} have a unique simple path between them consisting of EXACTLY K trails.

Input
The first line contains N and K.
Each of the next N - 1 lines contains two clearings connected by a trail.

Output
Print a single integer: the number of unordered pairs at distance K.

Constraints
2 <= N <= 180000
1 <= K < N""",
        samples=[
            "5 2\n1 2\n2 3\n2 4\n4 5\n",
        ],
        expected_samples=["4\n"],
        small_inputs=[
            "2 1\n1 2\n",
            gen_50(400, 7, 5001),
            gen_50(1500, 40, 5002, "caterpillar"),
        ],
        large_inputs=[
            gen_50(170000, 60000, 5011, "chain"),
            gen_50(180000, 15, 5012, "random"),
            gen_50(180000, 500, 5013, "caterpillar"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, K;
    cin >> n >> K;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), tmp_par(n + 1, 0), ord, cnt(K + 1, 0), sub_d;
    long long ans = 0;

    function<void(int)> solve = [&](int start) {
        ord.clear();
        ord.push_back(start);
        tmp_par[start] = 0;
        for (size_t i = 0; i < ord.size(); i++) {
            int u = ord[i];
            sz[u] = 1;
            for (int v : adj[u]) if (!dead[v] && v != tmp_par[u]) {
                tmp_par[v] = u;
                ord.push_back(v);
            }
        }
        int tot = (int)ord.size();
        if (tot <= K) return;
        for (int i = tot - 1; i >= 1; i--) sz[tmp_par[ord[i]]] += sz[ord[i]];
        int c = start;
        while (true) {
            int nxt = 0;
            for (int v : adj[c]) if (!dead[v] && v != tmp_par[c] && sz[v] > tot / 2) {
                nxt = v; break;
            }
            if (!nxt) break;
            c = nxt;
        }
        dead[c] = 1;
        cnt[0] = 1;
        int max_d = 0;
        for (int nb : adj[c]) if (!dead[nb]) {
            sub_d.clear();
            vector<tuple<int, int, int>> st = {{nb, c, 1}};
            while (!st.empty()) {
                auto [u, p, d] = st.back(); st.pop_back();
                sub_d.push_back(d);
                if (d < K) {
                    for (int v : adj[u]) if (!dead[v] && v != p) st.push_back({v, u, d + 1});
                }
            }
            for (int d : sub_d) ans += cnt[K - d];
            for (int d : sub_d) { cnt[d]++; max_d = max(max_d, d); }
        }
        for (int d = 0; d <= max_d; d++) cnt[d] = 0;
        for (int nb : adj[c]) if (!dead[nb]) solve(nb);
    };

    solve(1);
    cout << ans << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, K;
    cin >> n >> K;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    long long ans = 0;
    for (int s = 1; s <= n; s++) {
        vector<tuple<int, int, int>> st = {{s, 0, 0}};
        while (!st.empty()) {
            auto [u, p, d] = st.back(); st.pop_back();
            if (d == K) { if (u > s) ans++; continue; }
            for (int v : adj[u]) if (v != p) st.push_back({v, u, d + 1});
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_10())
