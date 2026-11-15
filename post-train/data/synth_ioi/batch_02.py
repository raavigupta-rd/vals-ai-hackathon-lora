"""Batch 2 (synth_06 .. synth_10): Trees, MST Bottleneck, Rerooting, Euler Tour, DSU-on-Tree, Virtual/Steiner Subtree."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def _rand_tree_edges(n: int, r: random.Random, kind: str = "random") -> list[tuple[int, int]]:
    edges = []
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    # Keep root 1 in place if needed
    idx1 = perm.index(1)
    perm[0], perm[idx1] = perm[idx1], perm[0]
    for i in range(1, n):
        if kind == "chain":
            p = perm[i - 1]
        elif kind == "star":
            p = perm[0]
        elif kind == "caterpillar":
            p = perm[max(0, i - r.randint(1, 3))]
        else:
            p = perm[r.randint(0, i - 1)]
        edges.append((p, perm[i]))
    return edges


def make_batch_02() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_06: MOUNTAIN COURIERS (1700) - Minimax Bottleneck Path (MST + LCA)
    # ------------------------------------------------------------------
    def gen_06(n: int, m: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        base = _rand_tree_edges(n, r, kind)
        lines = [f"{n} {m} {q}"]
        for u, v in base:
            lines.append(f"{u} {v} {r.randint(1, 10**9)}")
        for _ in range(m - (n - 1)):
            u, v = r.randint(1, n), r.randint(1, n)
            while u == v:
                v = r.randint(1, n)
            lines.append(f"{u} {v} {r.randint(1, 10**9)}")
        for _ in range(q):
            u, v = r.randint(1, n), r.randint(1, n)
            lines.append(f"{u} {v}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_06",
        title="Mountain Couriers",
        rating=1700,
        tags=["graphs", "mst", "trees", "lca", "dsu"],
        notes="Kruskal minimum spanning tree + binary lifting for maximum edge weight on tree paths.",
        story_and_spec="""A mountain province has N villages numbered 1 to N connected by M bidirectional roads. Road i connects villages U_i and V_i and crosses a pass of altitude W_i. The road network is connected (multiple roads may connect the same pair of villages).

You are asked Q courier queries. Each query specifies a start village A and a destination village B: find the minimum oxygen rating R such that a courier can travel from A to B using only roads with altitude at most R (if A = B, the answer is 0).

Input
The first line contains N, M, and Q.
Each of the next M lines contains U_i, V_i, and W_i.
Each of the next Q lines contains A and B.

Output
For each query print the minimum required altitude rating on its own line.

Constraints
2 <= N <= 200000
N - 1 <= M <= 200000
1 <= Q <= 200000
1 <= U_i, V_i <= N, U_i != V_i
1 <= W_i <= 10^9""",
        samples=[
            "4 5 3\n1 2 5\n2 3 8\n1 3 6\n3 4 4\n2 4 9\n1 4\n2 4\n3 3\n",
        ],
        expected_samples=["6\n6\n0\n"],
        small_inputs=[
            "2 2 2\n1 2 100\n1 2 50\n1 2\n1 1\n",
            gen_06(300, 600, 400, 601),
            gen_06(900, 1500, 900, 602, "chain"),
            gen_06(1000, 1800, 1000, 603),
        ],
        large_inputs=[
            gen_06(160000, 200000, 180000, 611),
            gen_06(180000, 200000, 180000, 612, "caterpillar"),
            gen_06(180000, 200000, 200000, 613, "chain"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n + 1), sz(n + 1, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a; sz[a] += sz[b];
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    struct Edge { int u, v, w; };
    vector<Edge> es(m);
    for (int i = 0; i < m; i++) cin >> es[i].u >> es[i].v >> es[i].w;
    sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) { return a.w < b.w; });

    DSU dsu(n);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (const auto& e : es) {
        if (dsu.unite(e.u, e.v)) {
            adj[e.u].push_back({e.v, e.w});
            adj[e.v].push_back({e.u, e.w});
        }
    }

    int LOG = 1;
    while ((1 << LOG) <= n) LOG++;
    vector<int> dep(n + 1, 0);
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<vector<int>> mx(LOG, vector<int>(n + 1, 0));

    vector<int> st = {1};
    dep[1] = 1;
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        for (auto [v, w] : adj[u]) {
            if (!dep[v]) {
                dep[v] = dep[u] + 1;
                up[0][v] = u;
                mx[0][v] = w;
                st.push_back(v);
            }
        }
    }
    for (int k = 1; k < LOG; k++) {
        for (int i = 1; i <= n; i++) {
            int mid = up[k - 1][i];
            up[k][i] = up[k - 1][mid];
            mx[k][i] = max(mx[k - 1][i], mx[k - 1][mid]);
        }
    }

    while (q--) {
        int u, v;
        cin >> u >> v;
        if (u == v) { cout << 0 << '\n'; continue; }
        int ans = 0;
        if (dep[u] < dep[v]) swap(u, v);
        int diff = dep[u] - dep[v];
        for (int k = 0; k < LOG; k++) if ((diff >> k) & 1) {
            ans = max(ans, mx[k][u]);
            u = up[k][u];
        }
        if (u != v) {
            for (int k = LOG - 1; k >= 0; k--) {
                if (up[k][u] != up[k][v]) {
                    ans = max({ans, mx[k][u], mx[k][v]});
                    u = up[k][u];
                    v = up[k][v];
                }
            }
            ans = max({ans, mx[0][u], mx[0][v]});
        }
        cout << ans << '\n';
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
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    const int INF = 2e9;
    vector<int> dist(n + 1);
    while (q--) {
        int s, t;
        cin >> s >> t;
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        dist[s] = 0;
        pq.push({0, s});
        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (u == t) break;
            if (d != dist[u]) continue;
            for (auto [v, w] : adj[u]) {
                int nd = max(d, w);
                if (nd < dist[v]) {
                    dist[v] = nd;
                    pq.push({nd, v});
                }
            }
        }
        cout << dist[t] << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_07: IMPERIAL BEACONS (1800) - Weighted Rerooting DP
    # ------------------------------------------------------------------
    def gen_07(n: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        w = [r.randint(1, 10**4) for _ in range(n)]
        es = _rand_tree_edges(n, r, kind)
        lines = [str(n), " ".join(map(str, w))]
        for u, v in es:
            lines.append(f"{u} {v} {r.randint(1, 10**4)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_07",
        title="Imperial Beacons",
        rating=1800,
        tags=["trees", "dp", "rerooting"],
        notes="Two-pass rerooting DP: ans[v] = ans[u] + (total_W - 2 * sub_W[v]) * edge_len.",
        story_and_spec="""An empire consists of N fortress cities numbered 1 to N connected by N - 1 bidirectional roads forming a tree. City i houses W_i messengers, and road j connects U_j and V_j with length L_j.

If the imperial capital is placed at city c, every messenger in every city v must travel along the unique simple path from v to c, covering distance dist(v, c). Compute for every city c = 1, 2, ..., N the total distance travelled by all messengers: sum_{v=1..N} W_v * dist(v, c).

Input
The first line contains N.
The second line contains W_1 ... W_N.
Each of the next N - 1 lines contains U_j, V_j, and L_j.

Output
Print N lines: the i-th line should contain the total messenger distance if city i is chosen as capital.

Constraints
1 <= N <= 200000
1 <= W_i <= 10000
1 <= L_j <= 10000""",
        samples=[
            "4\n2 1 3 4\n1 2 5\n2 3 2\n2 4 3\n",
        ],
        expected_samples=["58\n28\n36\n34\n"],
        small_inputs=[
            "1\n100\n",
            gen_07(300, 701),
            gen_07(1500, 702, "chain"),
            gen_07(1800, 703, "caterpillar"),
        ],
        large_inputs=[
            gen_07(180000, 711),
            gen_07(200000, 712, "chain"),
            gen_07(200000, 713, "star"),
            gen_07(200000, 714, "caterpillar"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> w(n + 1);
    long long total_w = 0;
    for (int i = 1; i <= n; i++) { cin >> w[i]; total_w += w[i]; }
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long l;
        cin >> u >> v >> l;
        adj[u].push_back({v, l});
        adj[v].push_back({u, l});
    }
    vector<int> par(n + 1, 0), ord;
    vector<long long> plen(n + 1, 0), dist1(n + 1, 0);
    ord.reserve(n);
    ord.push_back(1);
    par[1] = -1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, l] : adj[u]) if (v != par[u]) {
            par[v] = u;
            plen[v] = l;
            dist1[v] = dist1[u] + l;
            ord.push_back(v);
        }
    }
    vector<long long> subw = w, ans(n + 1, 0);
    for (int i = 1; i <= n; i++) ans[1] += w[i] * dist1[i];
    for (int i = n - 1; i >= 1; i--) {
        int u = ord[i];
        subw[par[u]] += subw[u];
    }
    for (int i = 1; i < n; i++) {
        int u = ord[i];
        ans[u] = ans[par[u]] + (total_w - 2LL * subw[u]) * plen[u];
    }
    for (int i = 1; i <= n; i++) cout << ans[i] << '\n';
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
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i];
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long l;
        cin >> u >> v >> l;
        adj[u].push_back({v, l});
        adj[v].push_back({u, l});
    }
    for (int root = 1; root <= n; root++) {
        long long total = 0;
        vector<tuple<int, int, long long>> st = {{root, 0, 0}};
        while (!st.empty()) {
            auto [u, p, d] = st.back(); st.pop_back();
            total += w[u] * d;
            for (auto [v, l] : adj[u]) if (v != p) st.push_back({v, u, d + l});
        }
        cout << total << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_08: ANCESTRAL ARCHIVE (2100) - Subtree Add + Path Sum
    # ------------------------------------------------------------------
    def gen_08(n: int, q: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        a = [r.randint(-10**5, 10**5) for _ in range(n)]
        es = _rand_tree_edges(n, r, kind)
        lines = [f"{n} {q}", " ".join(map(str, a))]
        for u, v in es:
            lines.append(f"{u} {v}")
        for i in range(q):
            if i < q - 1 and r.random() < 0.48:
                # Bias updates toward upper tree nodes as well as random nodes so subtrees are large
                u = es[r.randrange(min(len(es), 20))][0] if (es and r.random() < 0.5) else r.randint(1, n)
                lines.append(f"1 {u} {r.randint(-10**5, 10**5)}")
            else:
                lines.append(f"2 {r.randint(1, n)} {r.randint(1, n)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_08",
        title="Ancestral Archive",
        rating=2100,
        tags=["trees", "euler tour", "fenwick tree", "lca"],
        notes="Root-path sum f(u) changes on subtree(w) add x by x*depth[u] + x*(1-depth[w]) for tin[w]..tout[w]; maintain slope and constant BITs + LCA.",
        story_and_spec="""An academy of N scholars is organized as a rooted tree with scholar 1 as the grandmaster (root). Initially scholar i receives stipend A_i. Process Q events of two types:
- 1 U X: add X to the stipend of every scholar in the subtree of U (including U itself).
- 2 U V: report the sum of current stipends of all scholars on the unique simple path between U and V (inclusive).

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next N - 1 lines contains two scholars connected by a direct mentorship link.
Each of the next Q lines describes an event in one of the two formats above. At least one event is of type 2.

Output
For each event of type 2, print the path sum on its own line.

Constraints
1 <= N, Q <= 200000
-10^5 <= A_i, X <= 10^5
1 <= U, V <= N""",
        samples=[
            "5 4\n10 20 30 40 50\n1 2\n1 3\n2 4\n2 5\n2 4 3\n1 2 5\n2 4 5\n2 5 3\n",
        ],
        expected_samples=["100\n125\n120\n"],
        small_inputs=[
            "1 3\n7\n2 1 1\n1 1 -3\n2 1 1\n",
            gen_08(300, 400, 801),
            gen_08(1300, 1400, 802, "chain"),
            gen_08(1600, 1600, 803, "caterpillar"),
        ],
        large_inputs=[
            gen_08(200000, 200000, 812, "chain"),
            gen_08(200000, 200000, 813, "caterpillar"),
            gen_08(180000, 180000, 811),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct RangeBIT {
    int n;
    vector<long long> t;
    RangeBIT(int n) : n(n), t(n + 2, 0) {}
    void add_pt(int i, long long v) {
        for (; i <= n + 1; i += i & -i) t[i] += v;
    }
    void add_range(int l, int r, long long v) {
        add_pt(l, v);
        add_pt(r + 1, -v);
    }
    long long get(int i) const {
        long long s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int LOG = 1;
    while ((1 << LOG) <= n) LOG++;
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<int> par(n + 1, 0), dep(n + 1, 0), tin(n + 1, 0), tout(n + 1, 0);
    vector<long long> base_f(n + 1, 0);

    int timer = 0;
    // Iterative DFS for Euler tour
    vector<pair<int, int>> st = {{1, 0}};
    dep[1] = 1;
    base_f[1] = a[1];
    while (!st.empty()) {
        auto& [u, idx] = st.back();
        if (idx == 0) tin[u] = ++timer;
        if (idx < (int)adj[u].size()) {
            int v = adj[u][idx++];
            if (v == par[u]) continue;
            par[v] = u;
            up[0][v] = u;
            dep[v] = dep[u] + 1;
            base_f[v] = base_f[u] + a[v];
            st.push_back({v, 0});
        } else {
            tout[u] = timer;
            st.pop_back();
        }
    }
    for (int k = 1; k < LOG; k++)
        for (int i = 1; i <= n; i++)
            up[k][i] = up[k - 1][up[k - 1][i]];

    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int k = 0; k < LOG; k++) if ((d >> k) & 1) u = up[k][u];
        if (u == v) return u;
        for (int k = LOG - 1; k >= 0; k--)
            if (up[k][u] != up[k][v]) { u = up[k][u]; v = up[k][v]; }
        return up[0][u];
    };

    RangeBIT slope(n), cst(n);
    auto root_sum = [&](int u) -> long long {
        if (u == 0) return 0LL;
        return base_f[u] + slope.get(tin[u]) * dep[u] + cst.get(tin[u]);
    };

    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int u; long long x;
            cin >> u >> x;
            slope.add_range(tin[u], tout[u], x);
            cst.add_range(tin[u], tout[u], x * (1LL - dep[u]));
        } else {
            int u, v;
            cin >> u >> v;
            int w = lca(u, v);
            cout << (root_sum(u) + root_sum(v) - root_sum(w) - root_sum(par[w])) << '\n';
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
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1, 0), dep(n + 1, 0);
    vector<int> qv = {1};
    dep[1] = 1;
    for (int i = 0; i < n; i++) {
        int u = qv[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] + 1;
            qv.push_back(v);
        }
    }
    while (q--) {
        int t;
        cin >> t;
        if (t == 1) {
            int u; long long x;
            cin >> u >> x;
            vector<int> st = {u};
            while (!st.empty()) {
                int c = st.back(); st.pop_back();
                a[c] += x;
                for (int v : adj[c]) if (v != par[c]) st.push_back(v);
            }
        } else {
            int u, v;
            cin >> u >> v;
            long long s = 0;
            while (dep[u] > dep[v]) { s += a[u]; u = par[u]; }
            while (dep[v] > dep[u]) { s += a[v]; v = par[v]; }
            while (u != v) { s += a[u] + a[v]; u = par[u]; v = par[v]; }
            s += a[u];
            cout << s << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_09: FESTIVAL LANTERNS (2100) - DSU on Tree (Same Color, Even Dist)
    # ------------------------------------------------------------------
    def gen_09(n: int, cmax: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        c = [r.randint(1, cmax) for _ in range(n)]
        es = _rand_tree_edges(n, r, kind)
        lines = [str(n), " ".join(map(str, c))]
        for u, v in es:
            lines.append(f"{u} {v}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_09",
        title="Festival Lanterns",
        rating=2100,
        tags=["trees", "dsu on tree", "small to large"],
        notes="dist(x, y) is even iff depth[x] == depth[y] (mod 2). DSU-on-tree maintaining counts[color][parity] and running pair sum.",
        story_and_spec="""A festival park has N plazas numbered 1 to N arranged as a rooted tree with plaza 1 at the entrance. Plaza i hangs a lantern of colour C_i.

For every plaza u (1 <= u <= N), count how many unordered pairs of distinct plazas {x, y} inside the subtree of u have the SAME lantern colour (C_x = C_y) AND have an EVEN number of walkways on the simple path between x and y.

Input
The first line contains N.
The second line contains C_1 ... C_N.
Each of the next N - 1 lines contains two plazas connected by a walkway.

Output
Print N lines: the i-th line should contain the count of valid pairs inside the subtree of plaza i.

Constraints
1 <= N <= 180000
1 <= C_i <= N""",
        samples=[
            "6\n1 1 1 1 2 2\n1 2\n1 3\n2 4\n3 5\n5 6\n",
        ],
        expected_samples=["2\n0\n0\n0\n0\n0\n"],
        small_inputs=[
            "1\n1\n",
            "5\n2 2 2 2 2\n1 2\n2 3\n3 4\n4 5\n",
            gen_09(400, 10, 901),
            gen_09(1600, 30, 902, "chain"),
            gen_09(1800, 5, 903, "caterpillar"),
        ],
        large_inputs=[
            gen_09(180000, 5, 912, "chain"),
            gen_09(180000, 5000, 913, "caterpillar"),
            gen_09(160000, 50, 911),
            gen_09(180000, 2, 914),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> col(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), dep(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0);
    vector<int> tin(n + 1, 0), tout(n + 1, 0), ver(n + 1, 0);
    int timer = 0;
    {
        vector<pair<int, int>> st = {{1, 0}};
        while (!st.empty()) {
            auto& [u, i] = st.back();
            if (i == 0) {
                tin[u] = ++timer;
                ver[timer] = u;
            }
            if (i < (int)adj[u].size()) {
                int v = adj[u][i++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] ^ 1;
                st.push_back({v, 0});
            } else {
                tout[u] = timer;
                st.pop_back();
                if (par[u]) {
                    sz[par[u]] += sz[u];
                    if (sz[u] > sz[heavy[par[u]]]) heavy[par[u]] = u;
                }
            }
        }
    }

    vector<array<int, 2>> cnt(n + 1, {0, 0});
    long long cur_pairs = 0;
    vector<long long> ans(n + 1, 0);

    auto add_node = [&](int u) {
        int c = col[u], p = dep[u];
        cur_pairs += cnt[c][p];
        cnt[c][p]++;
    };
    auto rem_node = [&](int u) {
        int c = col[u], p = dep[u];
        cnt[c][p]--;
        cur_pairs -= cnt[c][p];
    };

    // Iterative DSU on tree
    struct Frame { int u; bool keep; int stage; int idx; };
    vector<Frame> st = {{1, true, 0, 0}};
    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.stage == 0) {
            if (f.idx < (int)adj[u].size()) {
                int v = adj[u][f.idx++];
                if (v == par[u] || v == heavy[u]) continue;
                st.push_back({v, false, 0, 0});
            } else {
                f.stage = 1;
                if (heavy[u]) st.push_back({heavy[u], true, 0, 0});
            }
        } else if (f.stage == 1) {
            for (int v : adj[u]) {
                if (v == par[u] || v == heavy[u]) continue;
                for (int t = tin[v]; t <= tout[v]; t++) add_node(ver[t]);
            }
            add_node(u);
            ans[u] = cur_pairs;
            if (!f.keep) {
                for (int t = tin[u]; t <= tout[u]; t++) rem_node(ver[t]);
            }
            st.pop_back();
        }
    }

    for (int i = 1; i <= n; i++) cout << ans[i] << '\n';
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
    vector<int> col(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1, 0), dep(n + 1, 0), ord = {1};
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            dep[v] = dep[u] ^ 1;
            ord.push_back(v);
        }
    }
    vector<array<int, 2>> cnt(n + 1, {0, 0});
    for (int root = 1; root <= n; root++) {
        vector<int> st = {root}, vis;
        long long pairs = 0;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            vis.push_back(u);
            pairs += cnt[col[u]][dep[u]];
            cnt[col[u]][dep[u]]++;
            for (int v : adj[u]) if (v != par[u]) st.push_back(v);
        }
        for (int u : vis) cnt[col[u]] = {0, 0};
        cout << pairs << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_10: EMERGENCY DEPOTS (2200) - Steiner Tree Length on Queries
    # ------------------------------------------------------------------
    def gen_10(n: int, q: int, total_k: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree_edges(n, r, kind)
        lines = [f"{n} {q}"]
        for u, v in es:
            lines.append(f"{u} {v} {r.randint(1, 10**6)}")
        rem = total_k
        for i in range(q):
            left_q = q - i
            k = max(1, min(n, rem - (left_q - 1) if i == q - 1 else r.randint(1, max(1, min(n, (2 * rem) // left_q)))))
            rem = max(left_q - 1, rem - k)
            picks = r.sample(range(1, n + 1), k)
            lines.append(f"{k} " + " ".join(map(str, picks)))
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_10",
        title="Emergency Depots",
        rating=2200,
        tags=["trees", "lca", "dfs order", "virtual tree"],
        notes="Sort query vertices by Euler tin; minimal spanning subtree weight is 0.5 * sum_i dist(v_i, v_{(i+1)%k}).",
        story_and_spec="""A kingdom has N cities numbered 1 to N connected by N - 1 bidirectional highways forming a tree, where highway j has length L_j.

You must evaluate Q emergency plans. Plan i specifies K_i distinct cities S_1, ..., S_{K_i}. Determine the minimum total length of highways that must be kept open so that all K_i specified cities are mutually reachable from one another (if K_i = 1, the answer is 0).

Input
The first line contains N and Q.
Each of the next N - 1 lines contains U_j, V_j, and L_j.
Each of the next Q lines begins with K_i followed by K_i distinct city numbers.
The sum of K_i over all Q plans does not exceed 300000.

Output
For each plan print the minimum total highway length on its own line.

Constraints
1 <= N, Q <= 180000
1 <= L_j <= 10^6
1 <= K_i <= N
sum of K_i <= 300000""",
        samples=[
            "6 3\n1 2 4\n1 3 5\n2 4 3\n2 5 6\n3 6 2\n3 4 5 6\n2 4 5\n1 3\n",
        ],
        expected_samples=["20\n9\n0\n"],
        small_inputs=[
            "2 2\n1 2 7\n1 1\n2 1 2\n",
            gen_10(300, 300, 1500, 1001),
            gen_10(1100, 900, 5000, 1002, "chain"),
            gen_10(1200, 1000, 6000, 1003, "caterpillar"),
        ],
        large_inputs=[
            gen_10(160000, 80000, 260000, 1011),
            gen_10(180000, 90000, 280000, 1012, "chain"),
            gen_10(180000, 90000, 280000, 1013, "caterpillar"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    vector<int> tin(n + 1, 0), first_pos(n + 1, 0), dep(n + 1, 0), par(n + 1, 0);
    vector<long long> dist(n + 1, 0);
    vector<int> euler;
    euler.reserve(2 * n);
    int timer = 0;

    vector<pair<int, int>> st = {{1, 0}};
    while (!st.empty()) {
        auto& [u, idx] = st.back();
        if (idx == 0) {
            tin[u] = ++timer;
            first_pos[u] = (int)euler.size();
            euler.push_back(u);
        }
        if (idx < (int)adj[u].size()) {
            auto [v, w] = adj[u][idx++];
            if (v == par[u]) continue;
            par[v] = u;
            dep[v] = dep[u] + 1;
            dist[v] = dist[u] + w;
            st.push_back({v, 0});
        } else {
            st.pop_back();
            if (!st.empty()) euler.push_back(st.back().first);
        }
    }

    int m = (int)euler.size();
    int LOG = __lg(m) + 1;
    vector<vector<int>> rmq(LOG, vector<int>(m));
    rmq[0] = euler;
    for (int k = 1; k < LOG; k++) {
        int half = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= m; i++) {
            int a = rmq[k - 1][i], b = rmq[k - 1][i + half];
            rmq[k][i] = (dep[a] < dep[b] ? a : b);
        }
    }
    auto lca = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        return dep[a] < dep[b] ? a : b;
    };
    auto tree_dist = [&](int u, int v) {
        int w = lca(u, v);
        return dist[u] + dist[v] - 2LL * dist[w];
    };

    while (q--) {
        int k;
        cin >> k;
        vector<int> v(k);
        for (int i = 0; i < k; i++) cin >> v[i];
        if (k <= 1) { cout << 0 << '\n'; continue; }
        sort(v.begin(), v.end(), [&](int a, int b) { return tin[a] < tin[b]; });
        long long sum = 0;
        for (int i = 0; i < k; i++) sum += tree_dist(v[i], v[(i + 1) % k]);
        cout << (sum / 2) << '\n';
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
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<int> par(n + 1, 0), ord = {1};
    vector<long long> pw(n + 1, 0);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, w] : adj[u]) if (v != par[u]) {
            par[v] = u;
            pw[v] = w;
            ord.push_back(v);
        }
    }
    vector<int> cnt(n + 1, 0);
    while (q--) {
        int k;
        cin >> k;
        fill(cnt.begin(), cnt.end(), 0);
        for (int i = 0; i < k; i++) {
            int x; cin >> x;
            cnt[x] = 1;
        }
        long long ans = 0;
        for (int i = n - 1; i >= 1; i--) {
            int u = ord[i];
            if (cnt[u] > 0 && cnt[u] < k) ans += pw[u];
            cnt[par[u]] += cnt[u];
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_02())
