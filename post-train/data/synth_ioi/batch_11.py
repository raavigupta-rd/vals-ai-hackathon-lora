"""Batch 11 (synth_51 .. synth_55): Graphs II — Two-Clan Multi-Source Dijkstra, Block-Cut Tree Chokepoints, Shortest-Path Edge Replacement, SegTree Range-Edge Dijkstra, Almost-Tree Shortest Paths."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_11() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_51: RIVAL OUTPOSTS (1900) - Top-2 Clan Multi-Source Dijkstra
    # ------------------------------------------------------------------
    def gen_51(n: int, m: int, cmax: int, seed: int, adv: bool = False) -> str:
        r = random.Random(seed)
        if adv:
            c = [1] * (n - 3) + [2] * 3
            lines = [f"{n} {m}", " ".join(map(str, c))]
            for i in range(1, n):
                lines.append(f"{i} {i + 1} {r.randint(1, 10**6)}")
            cur_m = n - 1
            while cur_m < m:
                u = r.randint(1, n - 3)
                v = min(n, u + r.randint(1, 5))
                if u != v:
                    lines.append(f"{u} {v} {r.randint(1, 10**6)}")
                    cur_m += 1
            return "\n".join(lines) + "\n"
        c = [r.randint(1, cmax) for _ in range(n)]
        lines = [f"{n} {m}", " ".join(map(str, c))]
        for i in range(1, min(n, m + 1)):
            lines.append(f"{i} {r.randint(1, i)} {r.randint(1, 10**8)}" if i > 1 else "")
        lines = [x for x in lines if x]
        cur_m = len(lines) - 2
        while cur_m < m:
            u, v = r.randint(1, n), r.randint(1, n)
            if u != v:
                lines.append(f"{u} {v} {r.randint(1, 10**8)}")
                cur_m += 1
        lines[0] = f"{n} {m}"
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_51",
        title="Rival Outposts",
        rating=1900,
        tags=["graphs", "shortest paths", "dijkstra", "multi-source"],
        notes="Multi-source Dijkstra starting from all N vertices simultaneously, keeping at each vertex the 2 shortest distances originating from DISTINCT clans.",
        story_and_spec="""A frontier region has N outposts numbered 1 to N connected by M bidirectional roads, where road i connects U_i and V_i with length W_i. Outpost u belongs to clan C_u.

For EVERY outpost u = 1, 2, ..., N, find the minimum road distance from u to any outpost v belonging to a DIFFERENT clan (C_v != C_u), or report -1 if no such outpost can be reached from u.

Input
The first line contains N and M.
The second line contains C_1 ... C_N.
Each of the next M lines contains U_i, V_i, and W_i.

Output
Print N space-separated integers on a single line: the answer for outposts 1, 2, ..., N.

Constraints
2 <= N <= 180000
1 <= M <= 200000
1 <= C_u <= N
1 <= U_i, V_i <= N, U_i != V_i
1 <= W_i <= 10^8""",
        samples=[
            "5 5\n1 1 2 2 3\n1 2 4\n2 3 7\n3 4 2\n4 5 5\n1 4 20\n",
        ],
        small_inputs=[
            "3 1\n1 1 2\n1 2 5\n",
            gen_51(300, 500, 4, 5101),
            gen_51(700, 1200, 2, 5102),
        ],
        large_inputs=[
            gen_51(160000, 190000, 3, 5111, adv=True),
            gen_51(180000, 200000, 50, 5112),
            gen_51(180000, 200000, 2, 5113),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<int> clan(n + 1);
    for (int i = 1; i <= n; i++) cin >> clan[i];
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    const long long INF = (1LL << 60);
    vector<array<long long, 2>> d(n + 1, {INF, INF});
    vector<array<int, 2>> src(n + 1, {0, 0});
    using T = tuple<long long, int, int>; // dist, node, clan
    priority_queue<T, vector<T>, greater<>> pq;

    for (int i = 1; i <= n; i++) {
        d[i][0] = 0;
        src[i][0] = clan[i];
        pq.push({0, i, clan[i]});
    }

    while (!pq.empty()) {
        auto [du, u, cu] = pq.top(); pq.pop();
        if (!((src[u][0] == cu && d[u][0] == du) || (src[u][1] == cu && d[u][1] == du))) continue;
        for (auto [v, w] : adj[u]) {
            long long nd = du + w;
            if (src[v][0] == cu) {
                if (nd < d[v][0]) { d[v][0] = nd; pq.push({nd, v, cu}); }
            } else if (nd < d[v][0]) {
                d[v][1] = d[v][0]; src[v][1] = src[v][0];
                d[v][0] = nd; src[v][0] = cu;
                pq.push({nd, v, cu});
            } else if (src[v][1] == cu) {
                if (nd < d[v][1]) { d[v][1] = nd; pq.push({nd, v, cu}); }
            } else if (nd < d[v][1]) {
                d[v][1] = nd; src[v][1] = cu;
                pq.push({nd, v, cu});
            }
        }
    }

    for (int i = 1; i <= n; i++) {
        long long ans = (src[i][0] != clan[i] ? d[i][0] : d[i][1]);
        cout << (ans >= INF ? -1LL : ans) << (i == n ? '\n' : ' ');
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
    int n, m;
    cin >> n >> m;
    vector<int> clan(n + 1);
    for (int i = 1; i <= n; i++) cin >> clan[i];
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1);
    for (int s = 1; s <= n; s++) {
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        dist[s] = 0;
        pq.push({0, s});
        long long ans = -1;
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du != dist[u]) continue;
            if (clan[u] != clan[s]) { ans = du; break; }
            for (auto [v, w] : adj[u]) if (du + w < dist[v]) {
                dist[v] = du + w;
                pq.push({dist[v], v});
            }
        }
        cout << ans << (s == n ? '\n' : ' ');
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_52: CHOKEPOINT CITIES (2200) - Block-Cut Tree + LCA
    # ------------------------------------------------------------------
    def gen_52(n: int, extra: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        perm = list(range(1, n + 1))
        r.shuffle(perm)
        used = set()
        edges = []
        for i in range(1, n):
            u = perm[max(0, i - r.randint(1, min(i, 5)))]
            v = perm[i]
            a, b = (u, v) if u < v else (v, u)
            used.add((a, b))
            edges.append((u, v))
        tries = 0
        while len(edges) < n - 1 + extra and tries < extra * 5:
            tries += 1
            idx = r.randrange(n)
            jdx = min(n - 1, max(0, idx + r.randint(-7, 7)))
            if idx == jdx:
                continue
            u, v = perm[idx], perm[jdx]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in used:
                used.add((a, b))
                edges.append((u, v))
        lines = [f"{n} {len(edges)} {q}"] + [f"{u} {v}" for u, v in edges]
        for _ in range(q):
            a = r.randint(1, n)
            b = r.randint(1, n)
            while b == a:
                b = r.randint(1, n)
            lines.append(f"{a} {b}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_52",
        title="Chokepoint Cities",
        rating=2200,
        tags=["graphs", "biconnected components", "block-cut tree", "trees", "lca"],
        notes="Construct bipartite Block-Cut Tree (original vertices 1..N + block nodes N+1..N+B); number of cut vertices strictly separating A != B is dist_BCT(A, B)/2 - 1.",
        story_and_spec="""A connected road network links N cities numbered 1 to N via M bidirectional roads (with no self-loops and at most one direct road between any pair of cities).

You are given Q strategic queries. Each query specifies two distinct cities A and B: count how many cities C (with C != A and C != B) have the property that if city C and all roads incident to C are removed, A and B become disconnected.

Input
The first line contains N, M, and Q.
Each of the next M lines contains U_i and V_i.
Each of the next Q lines contains two distinct integers A and B.

Output
For each query print the number of chokepoint cities between A and B on its own line.

Constraints
2 <= N <= 160000
N - 1 <= M <= 220000
1 <= Q <= 160000
1 <= U_i, V_i, A, B <= N, A != B""",
        samples=[
            "6 6 4\n1 2\n2 3\n1 3\n3 4\n4 5\n4 6\n1 2\n1 5\n5 6\n3 6\n",
        ],
        small_inputs=[
            "2 1 1\n1 2\n1 2\n",
            gen_52(120, 60, 150, 5201),
            gen_52(250, 120, 200, 5202),
        ],
        large_inputs=[
            gen_52(150000, 45000, 150000, 5211),
            gen_52(160000, 50000, 160000, 5212),
            gen_52(160000, 40000, 160000, 5213),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<int> tin(n + 1, 0), low(n + 1, 0), stk;
    vector<vector<int>> bct(2 * n + 5);
    int timer = 0, bcnt = 0;

    // Iterative Tarjan for vertex biconnected components
    struct Frame { int u, p, idx, child_v; bool returning; };
    vector<Frame> call = {{1, 0, 0, 0, false}};
    while (!call.empty()) {
        auto& f = call.back();
        int u = f.u;
        if (f.idx == 0 && !f.returning) {
            tin[u] = low[u] = ++timer;
            stk.push_back(u);
        }
        if (f.returning) {
            int v = f.child_v;
            f.returning = false;
            low[u] = min(low[u], low[v]);
            if (low[v] >= tin[u]) {
                int bnode = n + (++bcnt);
                if ((int)bct.size() <= bnode) bct.resize(bnode + 5);
                while (true) {
                    int x = stk.back(); stk.pop_back();
                    bct[bnode].push_back(x);
                    bct[x].push_back(bnode);
                    if (x == v) break;
                }
                bct[bnode].push_back(u);
                bct[u].push_back(bnode);
            }
        }
        bool pushed = false;
        while (f.idx < (int)g[u].size()) {
            int v = g[u][f.idx++];
            if (v == f.p) continue;
            if (tin[v]) {
                low[u] = min(low[u], tin[v]);
            } else {
                f.child_v = v;
                f.returning = true;
                call.push_back({v, u, 0, 0, false});
                pushed = true;
                break;
            }
        }
        if (!pushed) call.pop_back();
    }

    int V = n + bcnt;
    int LOG = 1;
    while ((1 << LOG) <= V + 1) LOG++;
    vector<vector<int>> up(LOG, vector<int>(V + 1, 1));
    vector<int> dep(V + 1, 0), qv = {1};
    dep[1] = 1;
    for (size_t i = 0; i < qv.size(); i++) {
        int u = qv[i];
        for (int v : bct[u]) if (!dep[v]) {
            dep[v] = dep[u] + 1;
            up[0][v] = u;
            qv.push_back(v);
        }
    }
    for (int b = 1; b < LOG; b++)
        for (int i = 1; i <= V; i++)
            up[b][i] = up[b - 1][up[b - 1][i]];

    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int b = 0; b < LOG; b++) if ((d >> b) & 1) u = up[b][u];
        if (u == v) return u;
        for (int b = LOG - 1; b >= 0; b--)
            if (up[b][u] != up[b][v]) { u = up[b][u]; v = up[b][v]; }
        return up[0][u];
    };

    while (q--) {
        int a, b;
        cin >> a >> b;
        int w = lca(a, b);
        int d = dep[a] + dep[b] - 2 * dep[w];
        cout << (d / 2 - 1) << '\n';
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
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    // Find one shortest BFS path A->B; any vertex separating A and B MUST lie on this path!
    // Test deleting each internal vertex on that path via BFS -> O(Q * (N + M)) worst-case on chains!
    vector<int> par(n + 1, 0), vis(n + 1, 0);
    int stamp = 0;
    while (q--) {
        int a, b; cin >> a >> b;
        ++stamp;
        vector<int> qv = {a};
        vis[a] = stamp;
        par[a] = 0;
        for (size_t i = 0; i < qv.size(); i++) {
            int u = qv[i];
            if (u == b) break;
            for (int v : g[u]) if (vis[v] != stamp) {
                vis[v] = stamp;
                par[v] = u;
                qv.push_back(v);
            }
        }
        vector<int> cands;
        for (int x = par[b]; x != a && x != 0; x = par[x]) cands.push_back(x);
        int ans = 0;
        for (int ban : cands) {
            ++stamp;
            vis[a] = stamp;
            vis[ban] = stamp;
            qv.assign(1, a);
            bool reach = false;
            for (size_t i = 0; i < qv.size(); i++) {
                int u = qv[i];
                if (u == b) { reach = true; break; }
                for (int v : g[u]) if (vis[v] != stamp) {
                    vis[v] = stamp;
                    qv.push_back(v);
                }
            }
            if (!reach) ans++;
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_53: DETOUR HIGHWAYS (2300) - Shortest-Path Single-Edge Replacement
    # ------------------------------------------------------------------
    def gen_53(n: int, m: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        # Build a long spine 1 -> 2 -> ... -> L -> N plus bypass chords so path length is ~n/3!
        spine_len = max(2, n // 2)
        edges = []
        for i in range(1, spine_len):
            edges.append((i, i + 1, r.randint(5, 20)))
        edges.append((spine_len, n, r.randint(5, 20)))
        for v in range(spine_len + 1, n):
            p = r.randint(1, v - 1)
            edges.append((p, v, r.randint(10, 200)))
        while len(edges) < m:
            u = r.randint(1, n)
            v = min(n, max(1, u + r.randint(-50, 50)))
            if u != v:
                edges.append((u, v, r.randint(20, 500)))
        lines = [f"{n} {len(edges)} {q}"] + [f"{u} {v} {w}" for u, v, w in edges]
        # Query mostly spine edges (1..spine_len) plus random edges
        for _ in range(q):
            eid = r.randint(1, spine_len) if r.random() < 0.8 else r.randint(1, len(edges))
            lines.append(str(eid))
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_53",
        title="Detour Highways",
        rating=2300,
        tags=["graphs", "shortest paths", "dijkstra", "segment tree", "trees"],
        notes="Malik-Mittal-Gupta replacement paths: fix one shortest path P from 1 to N, compute d1/L_attach and dN/R_attach, sweep bypass chords over P with a min-heap.",
        story_and_spec="""A national road network connects N cities numbered 1 to N with M bidirectional highways. Highway i (1 <= i <= M) connects U_i and V_i with travel time W_i. Initially city N is reachable from city 1.

You are asked Q independent contingency queries. Query j gives a highway index E_j: if ONLY highway E_j is closed for repairs (all other M - 1 highways remain open), what is the minimum travel time from city 1 to city N? Print -1 if closing highway E_j disconnects city 1 from city N.

Input
The first line contains N, M, and Q.
Each of the next M lines contains U_i, V_i, and W_i (for i = 1, ..., M).
Each of the next Q lines contains E_j.

Output
For each query print the shortest travel time from 1 to N without highway E_j (or -1) on its own line.

Constraints
2 <= N <= 140000
N - 1 <= M <= 200000
1 <= Q <= 140000
1 <= U_i, V_i <= N, U_i != V_i
1 <= W_i <= 10^8
1 <= E_j <= M""",
        samples=[
            "4 5 5\n1 2 2\n2 3 2\n3 4 2\n1 3 5\n2 4 6\n1\n2\n3\n4\n5\n",
        ],
        small_inputs=[
            "2 1 1\n1 2 10\n1\n",
            gen_53(180, 350, 180, 5301),
            gen_53(350, 700, 250, 5302),
        ],
        large_inputs=[
            gen_53(130000, 190000, 130000, 5311),
            gen_53(140000, 200000, 140000, 5312),
            gen_53(140000, 200000, 140000, 5313),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 60);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    struct Edge { int u, v; long long w; };
    vector<Edge> es(m + 1);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= m; i++) {
        cin >> es[i].u >> es[i].v >> es[i].w;
        adj[es[i].u].push_back({es[i].v, i});
        adj[es[i].v].push_back({es[i].u, i});
    }

    auto dijkstra = [&](int src, vector<long long>& d, vector<int>& par_u, vector<int>& par_e) {
        d.assign(n + 1, INF);
        par_u.assign(n + 1, 0);
        par_e.assign(n + 1, 0);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        d[src] = 0;
        pq.push({0, src});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du != d[u]) continue;
            for (auto [v, eid] : adj[u]) {
                long long nd = du + es[eid].w;
                if (nd < d[v]) {
                    d[v] = nd;
                    par_u[v] = u;
                    par_e[v] = eid;
                    pq.push({nd, v});
                }
            }
        }
    };

    vector<long long> d1, dN;
    vector<int> pu1, pe1, puN, peN;
    dijkstra(1, d1, pu1, pe1);
    dijkstra(n, dN, puN, peN);

    // Extract one shortest path P: v_0=1, v_1, ..., v_K=n
    vector<int> path_v, path_e;
    for (int cur = n; cur != 1; cur = pu1[cur]) {
        path_v.push_back(cur);
        path_e.push_back(pe1[cur]);
    }
    path_v.push_back(1);
    reverse(path_v.begin(), path_v.end());
    reverse(path_e.begin(), path_e.end());
    int K = (int)path_e.size();

    vector<int> pos_v(n + 1, -1), pos_e(m + 1, -1);
    for (int i = 0; i <= K; i++) pos_v[path_v[i]] = i;
    for (int i = 0; i < K; i++) pos_e[path_e[i]] = i; // 0..K-1

    // L_att[u]: earliest index on P of a shortest path from 1 to u (follow P first)
    vector<int> order1(n), orderN(n);
    iota(order1.begin(), order1.end(), 1);
    sort(order1.begin(), order1.end(), [&](int a, int b) { return d1[a] < d1[b]; });
    vector<int> L_att(n + 1, K + 1);
    for (int i = 0; i <= K; i++) L_att[path_v[i]] = i;
    for (int u : order1) if (d1[u] < INF) {
        for (auto [v, eid] : adj[u]) {
            if (pos_e[eid] == -1 && d1[u] + es[eid].w == d1[v]) {
                L_att[v] = min(L_att[v], L_att[u]);
            }
        }
    }

    iota(orderN.begin(), orderN.end(), 1);
    sort(orderN.begin(), orderN.end(), [&](int a, int b) { return dN[a] < dN[b]; });
    vector<int> R_att(n + 1, -1);
    for (int i = 0; i <= K; i++) R_att[path_v[i]] = i;
    for (int u : orderN) if (dN[u] < INF) {
        for (auto [v, eid] : adj[u]) {
            if (pos_e[eid] == -1 && dN[u] + es[eid].w == dN[v]) {
                R_att[v] = max(R_att[v], R_att[u]);
            }
        }
    }

    vector<vector<pair<long long, int>>> add_at(K + 1);
    for (int i = 1; i <= m; i++) {
        if (pos_e[i] != -1) continue;
        int u = es[i].u, v = es[i].v;
        long long w = es[i].w;
        for (int dir = 0; dir < 2; dir++) {
            if (d1[u] < INF && dN[v] < INF && L_att[u] < R_att[v]) {
                add_at[L_att[u]].push_back({d1[u] + w + dN[v], R_att[v]});
            }
            swap(u, v);
        }
    }

    vector<long long> repl(K, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    for (int i = 0; i < K; i++) {
        for (auto& pr : add_at[i]) pq.push(pr);
        while (!pq.empty() && pq.top().second <= i) pq.pop();
        if (!pq.empty()) repl[i] = pq.top().first;
    }

    long long base = d1[n];
    while (q--) {
        int eid; cin >> eid;
        int idx = pos_e[eid];
        long long ans = (idx == -1 ? base : repl[idx]);
        cout << (ans >= INF ? -1LL : ans) << '\n';
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
    struct Edge { int u, v; long long w; };
    vector<Edge> es(m + 1);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= m; i++) {
        cin >> es[i].u >> es[i].v >> es[i].w;
        adj[es[i].u].push_back({es[i].v, i});
        adj[es[i].v].push_back({es[i].u, i});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1);
    while (q--) {
        int ban; cin >> ban;
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        dist[1] = 0;
        pq.push({0, 1});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (u == n) break;
            if (du != dist[u]) continue;
            for (auto [v, eid] : adj[u]) if (eid != ban) {
                if (du + es[eid].w < dist[v]) {
                    dist[v] = du + es[eid].w;
                    pq.push({dist[v], v});
                }
            }
        }
        cout << (dist[n] >= INF ? -1LL : dist[n]) << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_54: CATAPULT EXPRESS (2100) - Segment-Tree Graph Range-Edge Dijkstra
    # ------------------------------------------------------------------
    def gen_54(n: int, m: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {m}"]
        for _ in range(m):
            u = r.randint(1, n)
            if r.random() < 0.7:
                l = r.randint(1, max(1, n // 4))
                rr = r.randint(max(l, 3 * n // 4), n)
            else:
                l, rr = r.randint(1, n), r.randint(1, n)
                if l > rr:
                    l, rr = rr, l
            lines.append(f"{u} {l} {rr} {r.randint(1, 10**8)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_54",
        title="Catapult Express",
        rating=2100,
        tags=["graphs", "shortest paths", "dijkstra", "segment tree"],
        notes="Build a downward segment tree of virtual vertices with 0-cost internal edges; each range [L, R] connects to O(log N) segment tree nodes.",
        story_and_spec="""Along a canyon stand N forts numbered 1 to N. There are M courier catapults: catapult i is stationed at fort U_i and can hurl a message capsule directly to ANY single fort x satisfying L_i <= x <= R_i in W_i minutes.

Find the minimum time in minutes for a message starting at fort 1 to reach every fort 1, 2, ..., N (or -1 if a fort is unreachable).

Input
The first line contains N and M.
Each of the next M lines contains U_i, L_i, R_i, and W_i.

Output
Print N space-separated integers on a single line: the minimum travel time from fort 1 to forts 1, 2, ..., N (or -1 if unreachable).

Constraints
1 <= N, M <= 150000
1 <= U_i <= N
1 <= L_i <= R_i <= N
1 <= W_i <= 10^8""",
        samples=[
            "5 4\n1 2 3 5\n3 4 5 4\n2 5 5 2\n5 1 1 10\n",
        ],
        small_inputs=[
            "3 1\n1 3 3 7\n",
            gen_54(300, 400, 5401),
            gen_54(900, 1000, 5402),
        ],
        large_inputs=[
            gen_54(130000, 140000, 5411),
            gen_54(150000, 150000, 5412),
            gen_54(150000, 150000, 5413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    int sz = 1;
    while (sz < n) sz <<= 1;
    // Tree nodes: 1 .. 2*sz - 1; leaf for fort i (1-based) is sz + (i - 1)
    int V = 2 * sz;
    vector<vector<pair<int, long long>>> adj(V);
    for (int u = 1; u < sz; u++) {
        adj[u].push_back({2 * u, 0});
        adj[u].push_back({2 * u + 1, 0});
    }
    for (int i = 0; i < m; i++) {
        int u, l, r; long long w;
        cin >> u >> l >> r >> w;
        int src = sz + u - 1;
        for (l += sz - 1, r += sz; l < r; l >>= 1, r >>= 1) {
            if (l & 1) adj[src].push_back({l++, w});
            if (r & 1) adj[src].push_back({--r, w});
        }
    }

    const long long INF = (1LL << 60);
    vector<long long> dist(V, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[sz] = 0;
    pq.push({0, sz});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) if (d + w < dist[v]) {
            dist[v] = d + w;
            pq.push({dist[v], v});
        }
    }
    for (int i = 1; i <= n; i++) {
        long long d = dist[sz + i - 1];
        cout << (d >= INF ? -1LL : d) << (i == n ? '\n' : ' ');
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
    int n, m;
    cin >> n >> m;
    struct Cat { int l, r; long long w; };
    vector<vector<Cat>> cats(n + 1);
    for (int i = 0; i < m; i++) {
        int u, l, r; long long w;
        cin >> u >> l >> r >> w;
        cats[u].push_back({l, r, w});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[1] = 0;
    pq.push({0, 1});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [l, r, w] : cats[u]) {
            long long nd = d + w;
            for (int v = l; v <= r; v++) if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << (dist[i] >= INF ? -1LL : dist[i]) << (i == n ? '\n' : ' ');
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_55: CHORDED TREE ROADS (2000) - Shortest Paths with M <= N + 20
    # ------------------------------------------------------------------
    def gen_55(n: int, extra: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        perm = list(range(1, n + 1))
        r.shuffle(perm)
        edges = []
        for i in range(1, n):
            p = perm[max(0, i - r.randint(1, 4))]
            edges.append((p, perm[i], r.randint(10, 10**6)))
        for _ in range(extra):
            u, v = r.randint(1, n), r.randint(1, n)
            while u == v:
                v = r.randint(1, n)
            edges.append((u, v, r.randint(1, 10**5)))
        lines = [f"{n} {len(edges)} {q}"] + [f"{u} {v} {w}" for u, v, w in edges]
        for _ in range(q):
            lines.append(f"{r.randint(1, n)} {r.randint(1, n)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_55",
        title="Chorded Tree Roads",
        rating=2000,
        tags=["graphs", "trees", "lca", "dijkstra", "shortest paths"],
        notes="Build a spanning tree for O(1) RMQ LCA tree distances, then run Dijkstra from one endpoint of each of the <= 21 non-tree edges.",
        story_and_spec="""A province has N towns numbered 1 to N connected by M bidirectional roads, where road i connects U_i and V_i with length W_i. The network is connected, and M <= N + 20 (that is, at most 21 extra roads beyond a spanning tree).

Answer Q shortest-path queries: each query specifies two towns A and B and asks for the minimum road distance between A and B.

Input
The first line contains N, M, and Q.
Each of the next M lines contains U_i, V_i, and W_i.
Each of the next Q lines contains A and B.

Output
For each query print the shortest distance between A and B on its own line.

Constraints
2 <= N <= 140000
N - 1 <= M <= N + 20
1 <= Q <= 140000
1 <= U_i, V_i, A, B <= N, U_i != V_i
1 <= W_i <= 10^8""",
        samples=[
            "5 6 3\n1 2 10\n2 3 10\n3 4 10\n4 5 10\n1 5 5\n2 4 3\n1 4\n2 5\n1 3\n",
        ],
        small_inputs=[
            "2 2 2\n1 2 10\n1 2 4\n1 2\n1 1\n",
            gen_55(300, 10, 400, 5501),
            gen_55(700, 18, 700, 5502),
        ],
        large_inputs=[
            gen_55(130000, 20, 130000, 5511),
            gen_55(140000, 21, 140000, 5512),
            gen_55(140000, 15, 140000, 5513),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n + 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        p[b] = a; return true;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<pair<int, long long>>> adj(n + 1), tadj(n + 1);
    DSU dsu(n);
    vector<int> specials;
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        if (dsu.unite(u, v)) {
            tadj[u].push_back({v, w});
            tadj[v].push_back({u, w});
        } else {
            specials.push_back(u);
        }
    }
    sort(specials.begin(), specials.end());
    specials.erase(unique(specials.begin(), specials.end()), specials.end());

    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
    vector<long long> tdist(n + 1, 0);
    euler.reserve(2 * n);
    {
        vector<pair<int, int>> st = {{1, 0}};
        while (!st.empty()) {
            auto& [u, idx] = st.back();
            if (idx == 0) {
                first_pos[u] = (int)euler.size();
                euler.push_back(u);
            }
            if (idx < (int)tadj[u].size()) {
                auto [v, w] = tadj[u][idx++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] + 1;
                tdist[v] = tdist[u] + w;
                st.push_back({v, 0});
            } else {
                st.pop_back();
                if (!st.empty()) euler.push_back(st.back().first);
            }
        }
    }
    int em = (int)euler.size(), LOG = __lg(em) + 1;
    vector<vector<int>> rmq(LOG, vector<int>(em));
    rmq[0] = euler;
    for (int k = 1; k < LOG; k++) {
        int h = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= em; i++) {
            int a = rmq[k - 1][i], b = rmq[k - 1][i + h];
            rmq[k][i] = dep[a] < dep[b] ? a : b;
        }
    }
    auto tree_d = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        int w = dep[a] < dep[b] ? a : b;
        return tdist[u] + tdist[v] - 2LL * tdist[w];
    };

    const long long INF = (1LL << 60);
    int S = (int)specials.size();
    vector<vector<long long>> sdist(S, vector<long long>(n + 1, INF));
    for (int i = 0; i < S; i++) {
        int src = specials[i];
        auto& d = sdist[i];
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        d[src] = 0;
        pq.push({0, src});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (du != d[u]) continue;
            for (auto [v, w] : adj[u]) if (du + w < d[v]) {
                d[v] = du + w;
                pq.push({d[v], v});
            }
        }
    }

    while (q--) {
        int a, b; cin >> a >> b;
        long long ans = tree_d(a, b);
        for (int i = 0; i < S; i++) ans = min(ans, sdist[i][a] + sdist[i][b]);
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
    vector<vector<pair<int, long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    const long long INF = (1LL << 60);
    vector<long long> dist(n + 1);
    while (q--) {
        int a, b; cin >> a >> b;
        fill(dist.begin(), dist.end(), INF);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
        dist[a] = 0;
        pq.push({0, a});
        while (!pq.empty()) {
            auto [du, u] = pq.top(); pq.pop();
            if (u == b) break;
            if (du != dist[u]) continue;
            for (auto [v, w] : adj[u]) if (du + w < dist[v]) {
                dist[v] = du + w;
                pq.push({dist[v], v});
            }
        }
        cout << dist[b] << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_11())
