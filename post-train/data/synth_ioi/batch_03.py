"""Batch 3 (synth_11 .. synth_15): Graphs — Virtual Hubs, Offline DSU, 2-ECC Bridges, Rollback Parity DSU, Functional Binary Lifting."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_03() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_11: GUILD PORTALS (1800) - Dijkstra with Virtual Hub Nodes
    # ------------------------------------------------------------------
    def gen_11(n: int, m: int, g: int, total_k: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {m} {g}"]
        for _ in range(m):
            u, v = r.randint(1, n), r.randint(1, n)
            while u == v:
                v = r.randint(1, n)
            lines.append(f"{u} {v} {r.randint(1, 10**8)}")
        rem = total_k
        for i in range(g):
            left = g - i
            sz = max(2, min(n, rem - 2 * (left - 1) if i == g - 1 else r.randint(2, max(2, min(n, (2 * rem) // left)))))
            rem = max(2 * (left - 1), rem - sz)
            members = r.sample(range(1, n + 1), sz)
            lines.append(f"{sz} {r.randint(1, 10**8)} " + " ".join(map(str, members)))
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_11",
        title="Guild Portals",
        rating=1800,
        tags=["graphs", "shortest paths", "dijkstra", "virtual nodes"],
        notes="Create one virtual hub node per guild: city -> hub costs C_g, hub -> city costs 0; run Dijkstra.",
        story_and_spec="""A realm has N cities numbered 1 to N, M bidirectional roads, and G mage guilds. Road i connects U_i and V_i with travel time W_i. Guild j operates portals in K_j distinct cities; a traveller at any city belonging to guild j may teleport directly to ANY other city belonging to guild j in time C_j.

Find the minimum travel time from city 1 to every city 1, 2, ..., N (or -1 if a city cannot be reached).

Input
The first line contains N, M, and G.
Each of the next M lines contains U_i, V_i, and W_i.
Each of the next G lines begins with K_j and C_j, followed by the K_j city numbers in guild j.
The sum of K_j over all guilds does not exceed 250000.

Output
Print N space-separated integers on a single line: the minimum travel time from city 1 to cities 1, 2, ..., N (or -1 if unreachable).

Constraints
2 <= N <= 180000
0 <= M <= 180000
0 <= G <= 100000
2 <= K_j <= N, sum of K_j <= 250000
1 <= W_i, C_j <= 10^8""",
        samples=[
            "5 2 2\n1 2 10\n4 5 3\n3 4 1 3 4\n2 20 2 5\n",
        ],
        expected_samples=["0 10 4 4 7\n"],
        small_inputs=[
            "3 0 1\n2 5 1 3\n",
            gen_11(200, 250, 15, 400, 1101),
            gen_11(600, 700, 20, 900, 1102),
            gen_11(800, 900, 8, 1000, 1103),
        ],
        large_inputs=[
            # Huge guilds of size 60k so O(sum K^2) immediately blows up
            (lambda: (
                lambda n, r: (
                    f"{n} {n} 4\n"
                    + "\n".join(f"{i} {(i % n) + 1} {r.randint(100, 10**7)}" for i in range(1, n + 1)) + "\n"
                    + "\n".join(
                        f"60000 {r.randint(1000, 10**7)} " + " ".join(map(str, r.sample(range(1, n + 1), 60000)))
                        for _ in range(4)
                    ) + "\n"
                )
            )(160000, random.Random(1111)))(),
            gen_11(160000, 180000, 30, 240000, 1112),
            gen_11(180000, 180000, 15, 240000, 1113),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, g;
    cin >> n >> m >> g;
    int V = n + g;
    vector<vector<pair<int, long long>>> adj(V + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    for (int j = 1; j <= g; j++) {
        int k; long long c;
        cin >> k >> c;
        int hub = n + j;
        for (int t = 0; t < k; t++) {
            int u; cin >> u;
            adj[u].push_back({hub, c});
            adj[hub].push_back({u, 0});
        }
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(V + 1, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[1] = 0;
    pq.push({0, 1});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (d + w < dist[v]) {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << (dist[i] == INF ? -1LL : dist[i]) << (i == n ? '\n' : ' ');
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
    int n, m, g;
    cin >> n >> m >> g;
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    for (int j = 0; j < g; j++) {
        int k; long long c;
        cin >> k >> c;
        vector<int> mem(k);
        for (int i = 0; i < k; i++) cin >> mem[i];
        for (int a = 0; a < k; a++) {
            for (int b = a + 1; b < k; b++) {
                adj[mem[a]].push_back({mem[b], c});
                adj[mem[b]].push_back({mem[a], c});
            }
        }
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[1] = 0;
    pq.push({0, 1});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (d + w < dist[v]) {
                dist[v] = d + w;
                pq.push({dist[v], v});
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << (dist[i] == INF ? -1LL : dist[i]) << (i == n ? '\n' : ' ');
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_12: SUNKEN CAUSEWAYS (1700) - Offline DSU by Flood Level
    # ------------------------------------------------------------------
    def gen_12(n: int, m: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {m} {q}"]
        for _ in range(m):
            u, v = r.randint(1, n), r.randint(1, n)
            while u == v:
                v = r.randint(1, n)
            lines.append(f"{u} {v} {r.randint(1, 10**9)}")
        for _ in range(q):
            lines.append(f"{r.randint(1, n)} {r.randint(0, 10**9)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_12",
        title="Sunken Causeways",
        rating=1700,
        tags=["graphs", "dsu", "offline", "sorting"],
        notes="Sort causeways and queries in descending order of height/tide; maintain DSU component sizes.",
        story_and_spec="""An archipelago has N islands numbered 1 to N and M bidirectional stone causeways. Causeway i connects islands U_i and V_i at deck elevation H_i. When the sea tide rises to level T, every causeway with H_i <= T is submerged and impassable, while causeways with H_i > T remain open.

You are given Q tide forecasts. Each forecast specifies a starting island S and a tide level T: count how many islands (including S itself) can be reached from S using only open causeways.

Input
The first line contains N, M, and Q.
Each of the next M lines contains U_i, V_i, and H_i.
Each of the next Q lines contains S and T.

Output
For each forecast print the number of reachable islands on its own line.

Constraints
1 <= N <= 200000
0 <= M <= 200000
1 <= Q <= 200000
1 <= U_i, V_i, S <= N
1 <= H_i <= 10^9
0 <= T <= 10^9""",
        samples=[
            "5 5 4\n1 2 5\n2 3 8\n3 4 3\n4 5 9\n1 5 2\n2 4\n2 8\n4 3\n1 1\n",
        ],
        expected_samples=["3\n1\n2\n5\n"],
        small_inputs=[
            "3 0 2\n1 0\n2 10\n",
            gen_12(400, 800, 500, 1201),
            gen_12(1200, 1800, 1200, 1202),
            gen_12(1400, 2000, 1400, 1203),
        ],
        large_inputs=[
            gen_12(180000, 200000, 180000, 1211),
            gen_12(200000, 200000, 200000, 1212),
            # Dense connected low-threshold queries so BFS scans whole graph
            (lambda: (
                lambda n, m, q, r: (
                    f"{n} {m} {q}\n"
                    + "\n".join(f"{i} {i+1} {r.randint(500, 10**9)}" for i in range(1, n)) + "\n"
                    + "\n".join(f"{r.randint(1, n)} {r.randint(1, n)} {r.randint(500, 10**9)}" for _ in range(m - (n - 1))) + "\n"
                    + "\n".join(f"{r.randint(1, n)} {r.randint(0, 400)}" for _ in range(q)) + "\n"
                )
            )(180000, 200000, 180000, random.Random(1213)))(),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n + 1), sz(n + 1, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a; sz[a] += sz[b];
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    struct Edge { int u, v, h; };
    vector<Edge> es(m);
    for (int i = 0; i < m; i++) cin >> es[i].u >> es[i].v >> es[i].h;
    struct Qry { int s, t, id; };
    vector<Qry> qs(q);
    for (int i = 0; i < q; i++) { cin >> qs[i].s >> qs[i].t; qs[i].id = i; }

    sort(es.begin(), es.end(), [](const Edge& a, const Edge& b) { return a.h > b.h; });
    sort(qs.begin(), qs.end(), [](const Qry& a, const Qry& b) { return a.t > b.t; });

    DSU dsu(n);
    vector<int> ans(q);
    int p = 0;
    for (const auto& qr : qs) {
        while (p < m && es[p].h > qr.t) {
            dsu.unite(es[p].u, es[p].v);
            p++;
        }
        ans[qr.id] = dsu.sz[dsu.find(qr.s)];
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
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v, h;
        cin >> u >> v >> h;
        adj[u].push_back({v, h});
        adj[v].push_back({u, h});
    }
    vector<int> vis(n + 1, 0);
    for (int step = 1; step <= q; step++) {
        int s, t;
        cin >> s >> t;
        vector<int> st = {s};
        vis[s] = step;
        int cnt = 0;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            cnt++;
            for (auto [v, h] : adj[u]) if (h > t && vis[v] != step) {
                vis[v] = step;
                st.push_back(v);
            }
        }
        cout << cnt << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_13: CRITICAL AQUEDUCTS (2000) - Bridges on 2-ECC Tree + LCA
    # ------------------------------------------------------------------
    def gen_13(n: int, extra: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        edges = []
        perm = list(range(1, n + 1))
        r.shuffle(perm)
        for i in range(1, n):
            p = perm[max(0, i - r.randint(1, max(1, min(i, 6))))]
            edges.append((p, perm[i]))
        for _ in range(extra):
            idx = r.randrange(n)
            jdx = min(n - 1, max(0, idx + r.randint(-8, 8)))
            if idx != jdx:
                edges.append((perm[idx], perm[jdx]))
        r.shuffle(edges)
        lines = [f"{n} {len(edges)} {q}"]
        for u, v in edges:
            lines.append(f"{u} {v}")
        for _ in range(q):
            lines.append(f"{r.randint(1, n)} {r.randint(1, n)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_13",
        title="Critical Aqueducts",
        rating=2000,
        tags=["graphs", "bridges", "2-ecc", "trees", "lca"],
        notes="Find bridges via Tarjan (tracking edge IDs for parallel pipes), contract 2-edge-connected components into a tree, answer queries via LCA distance.",
        story_and_spec="""A city water network consists of N reservoirs numbered 1 to N connected by M bidirectional pipes. The network is connected, and multiple parallel pipes may directly connect the same pair of reservoirs.

You are asked Q vulnerability queries. Each query specifies two reservoirs A and B: count how many single pipes have the property that removing ONLY that pipe would disconnect reservoir A from reservoir B.

Input
The first line contains N, M, and Q.
Each of the next M lines contains two integers U_i and V_i (U_i != V_i), the endpoints of pipe i.
Each of the next Q lines contains A and B.

Output
For each query print the number of critical pipes between A and B on its own line.

Constraints
2 <= N <= 180000
N - 1 <= M <= 220000
1 <= Q <= 180000
1 <= U_i, V_i, A, B <= N, U_i != V_i""",
        samples=[
            "6 7 4\n1 2\n2 3\n1 3\n3 4\n4 5\n5 6\n5 6\n1 6\n1 2\n4 6\n3 5\n",
        ],
        expected_samples=["2\n0\n1\n2\n"],
        small_inputs=[
            "2 2 2\n1 2\n1 2\n1 2\n1 1\n",
            "3 2 2\n1 2\n2 3\n1 3\n2 3\n",
            gen_13(250, 120, 300, 1301),
            gen_13(500, 250, 500, 1302),
        ],
        large_inputs=[
            gen_13(160000, 40000, 160000, 1311),
            gen_13(180000, 30000, 180000, 1312),
            gen_13(180000, 40000, 180000, 1313),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    vector<pair<int, int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }

    vector<int> tin(n + 1, 0), low(n + 1, 0);
    vector<char> is_bridge(m, 0);
    int timer = 0;

    // Iterative Tarjan DFS
    struct Frame { int u, pe, idx; };
    vector<Frame> st = {{1, -1, 0}};
    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.idx == 0) tin[u] = low[u] = ++timer;
        if (f.idx < (int)adj[u].size()) {
            auto [v, eid] = adj[u][f.idx++];
            if (eid == f.pe) continue;
            if (tin[v]) {
                low[u] = min(low[u], tin[v]);
            } else {
                st.push_back({v, eid, 0});
            }
        } else {
            int pe = f.pe;
            st.pop_back();
            if (!st.empty()) {
                int p = st.back().u;
                low[p] = min(low[p], low[u]);
                if (low[u] > tin[p]) is_bridge[pe] = 1;
            }
        }
    }

    vector<int> comp(n + 1, 0);
    int ccnt = 0;
    for (int i = 1; i <= n; i++) if (!comp[i]) {
        ++ccnt;
        vector<int> qv = {i};
        comp[i] = ccnt;
        while (!qv.empty()) {
            int u = qv.back(); qv.pop_back();
            for (auto [v, eid] : adj[u]) {
                if (!is_bridge[eid] && !comp[v]) {
                    comp[v] = ccnt;
                    qv.push_back(v);
                }
            }
        }
    }

    vector<vector<int>> tree(ccnt + 1);
    for (int i = 0; i < m; i++) if (is_bridge[i]) {
        int a = comp[edges[i].first], b = comp[edges[i].second];
        tree[a].push_back(b);
        tree[b].push_back(a);
    }

    int LOG = 1;
    while ((1 << LOG) <= ccnt + 1) LOG++;
    vector<vector<int>> up(LOG, vector<int>(ccnt + 1, 1));
    vector<int> dep(ccnt + 1, 0);
    {
        vector<int> qv = {1};
        dep[1] = 1;
        for (int i = 0; i < (int)qv.size(); i++) {
            int u = qv[i];
            for (int v : tree[u]) if (!dep[v]) {
                dep[v] = dep[u] + 1;
                up[0][v] = u;
                qv.push_back(v);
            }
        }
    }
    for (int k = 1; k < LOG; k++)
        for (int i = 1; i <= ccnt; i++)
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

    while (q--) {
        int a, b;
        cin >> a >> b;
        int ca = comp[a], cb = comp[b];
        int w = lca(ca, cb);
        cout << (dep[ca] + dep[cb] - 2 * dep[w]) << '\n';
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
        int u, v;
        cin >> u >> v;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }
    // For each query, find ONE path A->B, then test removing each edge on that path!
    // Still O(Q * (N + M) + Q * path_len * M) -> make small N,M,Q <= 500 or O(Q*(N+M)) via per-query Tarjan!
    // Per-query DFS from A tracking bridges on the ancestor chain to B is O(N + M) per query => O(Q*(N+M))!
    vector<int> tin(n + 1), low(n + 1), par_e(n + 1), par_u(n + 1);
    while (q--) {
        int a, b;
        cin >> a >> b;
        if (a == b) { cout << 0 << '\n'; continue; }
        fill(tin.begin(), tin.end(), 0);
        int timer = 0;
        function<void(int, int)> dfs = [&](int u, int pe) {
            tin[u] = low[u] = ++timer;
            for (auto [v, eid] : adj[u]) {
                if (eid == pe) continue;
                if (tin[v]) low[u] = min(low[u], tin[v]);
                else {
                    par_u[v] = u;
                    par_e[v] = eid;
                    dfs(v, eid);
                    low[u] = min(low[u], low[v]);
                }
            }
        };
        dfs(a, -1);
        int ans = 0, cur = b;
        while (cur != a) {
            int p = par_u[cur];
            if (low[cur] > tin[p]) ans++;
            cur = p;
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_14: FACTION TRUCE (2300) - Dynamic Bipartiteness (SegTree + Rollback DSU)
    # ------------------------------------------------------------------
    def gen_14(n: int, m: int, T: int, seed: int) -> str:
        r = random.Random(seed)
        col = [r.randint(0, 1) for _ in range(n + 1)]
        lines = [f"{n} {m} {T}"]
        for i in range(m):
            l, rr = r.randint(1, T), r.randint(1, T)
            if l > rr:
                l, rr = rr, l
            u = r.randint(1, n)
            v = r.randint(1, n)
            while v == u:
                v = r.randint(1, n)
            # Mostly bipartite edges, plus occasional short odd-cycle spoilers
            if r.random() < 0.85 and n >= 4:
                tries = 0
                while col[u] == col[v] and tries < 10:
                    v = r.randint(1, n)
                    tries += 1
            else:
                rr = min(T, l + max(1, T // 6))
            lines.append(f"{u} {v} {l} {rr}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_14",
        title="Faction Truce",
        rating=2300,
        tags=["graphs", "divide and conquer", "dsu", "bipartite", "segment tree"],
        notes="Offline segment tree over time [1..T] + rollback DSU with parity bits to test bipartiteness on every day.",
        story_and_spec="""Along a frontier stand N forts numbered 1 to N. Over a campaign of T days (numbered 1 to T), each fort must be assigned every day to either the Dawn Watch or the Dusk Watch.

There are M rivalry pacts. Pact i specifies two forts U_i and V_i and a range of days [L_i, R_i]: on every day t with L_i <= t <= R_i, forts U_i and V_i refuse to serve on the same watch and must be assigned to OPPOSITE watches.

For each day t = 1, 2, ..., T, determine whether there exists a valid assignment of all N forts to the two watches satisfying all pacts active on day t.

Input
The first line contains N, M, and T.
Each of the next M lines contains U_i, V_i, L_i, and R_i.

Output
Print T lines: on line t print YES if a valid assignment exists on day t, or NO otherwise.

Constraints
2 <= N <= 150000
1 <= M, T <= 150000
1 <= U_i, V_i <= N, U_i != V_i
1 <= L_i <= R_i <= T""",
        samples=[
            "4 4 4\n1 2 1 4\n2 3 1 3\n1 3 2 3\n3 4 3 4\n",
        ],
        expected_samples=["YES\nNO\nNO\nYES\n"],
        small_inputs=[
            "2 1 2\n1 2 1 1\n",
            gen_14(200, 350, 300, 1401),
            gen_14(800, 1100, 900, 1402),
            gen_14(1000, 1200, 1000, 1403),
        ],
        large_inputs=[
            gen_14(140000, 150000, 150000, 1411),
            gen_14(150000, 150000, 150000, 1412),
            gen_14(150000, 150000, 150000, 1413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct RollbackParityDSU {
    vector<int> p, sz, xr;
    vector<pair<int, int>> hist; // (child, old_sz_parent)

    RollbackParityDSU(int n) : p(n + 1), sz(n + 1, 1), xr(n + 1, 0) {
        iota(p.begin(), p.end(), 0);
    }
    pair<int, int> find(int x) const {
        int par = 0;
        while (p[x] != x) {
            par ^= xr[x];
            x = p[x];
        }
        return {x, par};
    }
    bool add_edge(int u, int v) {
        auto [ru, pu] = find(u);
        auto [rv, pv] = find(v);
        if (ru == rv) return (pu ^ pv) == 1;
        if (sz[ru] < sz[rv]) { swap(ru, rv); swap(pu, pv); }
        hist.push_back({rv, sz[ru]});
        p[rv] = ru;
        xr[rv] = pu ^ pv ^ 1;
        sz[ru] += sz[rv];
        return true;
    }
    void rollback(size_t snap) {
        while (hist.size() > snap) {
            auto [rv, old_sz] = hist.back();
            hist.pop_back();
            int ru = p[rv];
            sz[ru] = old_sz;
            p[rv] = rv;
            xr[rv] = 0;
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, T;
    cin >> n >> m >> T;
    vector<vector<pair<int, int>>> seg(4 * T + 4);

    function<void(int, int, int, int, int, pair<int, int>)> insert_edge =
        [&](int u, int l, int r, int ql, int qr, pair<int, int> e) {
            if (ql <= l && r <= qr) {
                seg[u].push_back(e);
                return;
            }
            int mid = (l + r) >> 1;
            if (ql <= mid) insert_edge(2 * u, l, mid, ql, qr, e);
            if (qr > mid) insert_edge(2 * u + 1, mid + 1, r, ql, qr, e);
        };

    for (int i = 0; i < m; i++) {
        int u, v, l, r;
        cin >> u >> v >> l >> r;
        insert_edge(1, 1, T, l, r, {u, v});
    }

    RollbackParityDSU dsu(n);
    vector<char> ans(T + 1, 0);

    function<void(int, int, int, bool)> solve = [&](int u, int l, int r, bool ok) {
        size_t snap = dsu.hist.size();
        if (ok) {
            for (auto [a, b] : seg[u]) {
                if (!dsu.add_edge(a, b)) { ok = false; break; }
            }
        }
        if (l == r) {
            ans[l] = ok ? 1 : 0;
        } else {
            int mid = (l + r) >> 1;
            solve(2 * u, l, mid, ok);
            solve(2 * u + 1, mid + 1, r, ok);
        }
        dsu.rollback(snap);
    };

    solve(1, 1, T, true);
    for (int t = 1; t <= T; t++) cout << (ans[t] ? "YES\n" : "NO\n");
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, T;
    cin >> n >> m >> T;
    struct Pact { int u, v, l, r; };
    vector<Pact> ps(m);
    for (int i = 0; i < m; i++) cin >> ps[i].u >> ps[i].v >> ps[i].l >> ps[i].r;

    vector<vector<int>> adj(n + 1);
    vector<int> col(n + 1, -1), touched;
    for (int t = 1; t <= T; t++) {
        touched.clear();
        for (const auto& p : ps) if (p.l <= t && t <= p.r) {
            if (adj[p.u].empty()) touched.push_back(p.u);
            if (adj[p.v].empty()) touched.push_back(p.v);
            adj[p.u].push_back(p.v);
            adj[p.v].push_back(p.u);
        }
        bool ok = true;
        for (int s : touched) if (col[s] == -1 && ok) {
            col[s] = 0;
            vector<int> st = {s};
            while (!st.empty() && ok) {
                int u = st.back(); st.pop_back();
                for (int v : adj[u]) {
                    if (col[v] == -1) { col[v] = col[u] ^ 1; st.push_back(v); }
                    else if (col[v] == col[u]) { ok = false; break; }
                }
            }
        }
        for (int u : touched) { adj[u].clear(); col[u] = -1; }
        cout << (ok ? "YES\n" : "NO\n");
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_15: CONVOY ESCORT (1800) - Functional Graph 60-bit Binary Lifting
    # ------------------------------------------------------------------
    def gen_15(n: int, q: int, kmax: int, seed: int) -> str:
        r = random.Random(seed)
        nxt = [r.randint(1, n) for _ in range(n)]
        w = [r.randint(1, 10**6) for _ in range(n)]
        lines = [f"{n} {q}", " ".join(map(str, nxt)), " ".join(map(str, w))]
        for _ in range(q):
            lines.append(f"{r.randint(1, n)} {r.randint(1, kmax)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_15",
        title="Convoy Escort",
        rating=1800,
        tags=["graphs", "functional graph", "binary lifting"],
        notes="60-level binary lifting table storing destination and sum of weights over 2^b steps.",
        story_and_spec="""A desert supply network has N outposts numbered 1 to N. Outpost i sends all departing convoys along a single one-way road directly to outpost P_i (possibly P_i = i), taking W_i hours.

You are given Q dispatch orders. Each order specifies a starting outpost S and a step count K: if a convoy starts at S and traverses exactly K roads one after another, determine which outpost it ends at AND the total hours spent travelling.

Input
The first line contains N and Q.
The second line contains P_1 ... P_N.
The third line contains W_1 ... W_N.
Each of the next Q lines contains S and K.

Output
For each dispatch order print two space-separated integers on its own line: the final outpost and the total travel time.

Constraints
1 <= N, Q <= 200000
1 <= P_i, S <= N
1 <= W_i <= 10^6
1 <= K <= 10^15""",
        samples=[
            "5 3\n2 3 4 2 3\n5 10 2 8 7\n1 1\n1 4\n5 6\n",
        ],
        expected_samples=["2 5\n2 25\n2 37\n"],
        small_inputs=[
            "1 2\n1\n9\n1 1\n1 100\n",
            gen_15(400, 600, 1500, 1501),
            gen_15(1200, 1500, 1800, 1502),
            gen_15(1500, 1500, 2000, 1503),
        ],
        large_inputs=[
            gen_15(180000, 180000, 10**15, 1511),
            gen_15(200000, 200000, 10**15, 1512),
            gen_15(200000, 200000, 10**15, 1513),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int LOG = 51; // 2^50 > 10^15
    vector<array<int, LOG>> up(n + 1);
    vector<array<long long, LOG>> sm(n + 1);
    for (int i = 1; i <= n; i++) cin >> up[i][0];
    for (int i = 1; i <= n; i++) cin >> sm[i][0];

    for (int b = 1; b < LOG; b++) {
        for (int i = 1; i <= n; i++) {
            int mid = up[i][b - 1];
            up[i][b] = up[mid][b - 1];
            sm[i][b] = sm[i][b - 1] + sm[mid][b - 1];
        }
    }
    while (q--) {
        int u; long long k;
        cin >> u >> k;
        long long total = 0;
        for (int b = 0; b < LOG; b++) if ((k >> b) & 1LL) {
            total += sm[u][b];
            u = up[u][b];
        }
        cout << u << ' ' << total << '\n';
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
    vector<int> p(n + 1);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> p[i];
    for (int i = 1; i <= n; i++) cin >> w[i];
    while (q--) {
        int u; long long k;
        cin >> u >> k;
        long long total = 0;
        for (long long step = 0; step < k; step++) {
            total += w[u];
            u = p[u];
        }
        cout << u << ' ' << total << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_03())
