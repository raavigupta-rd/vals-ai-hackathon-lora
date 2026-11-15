"""Batch 18 (synth_86 .. synth_90): Grandmaster Wave II (CF 2700-2800) — Pairwise Weighted Virtual-Tree Min-Cut, Lengauer-Tarjan Dominator Tree on Cyclic Digraphs, Offline Dynamic Connectivity (Time SegTree + Rollback DSU), Interval Project Selection Closure, Generalized Hall/Min-Cut Prefix-Suffix Bipartite Flow."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def _rand_tree_weighted(n: int, r: random.Random, kind: str = "random") -> list[tuple[int, int, int]]:
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    es = []
    for i in range(1, n):
        if kind == "chain":
            p = perm[i - 1]
        elif kind == "caterpillar":
            p = perm[max(0, i - r.randint(1, 3))]
        else:
            p = perm[r.randint(0, i - 1)]
        es.append((p, perm[i], r.randint(1, 10**8)))
    return es


def make_batch_18() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_86: EMERGENCY RELAY CUT (2800) - Weighted Pairwise Separation on Virtual Tree
    # ------------------------------------------------------------------
    def gen_86(n: int, q: int, sum_k: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree_weighted(n, r, kind)
        lines = [f"{n} {q}"] + [f"{u} {v} {w}" for u, v, w in es]
        rem = sum_k
        for i in range(q):
            q_left = q - i
            avg = max(2, rem // q_left)
            k = min(n, max(2, r.randint(2, min(n, max(2, 2 * avg - 1)))))
            if i == q - 1:
                k = min(n, max(2, rem))
            rem = max(2 * (q_left - 1), rem - k)
            picks = r.sample(range(1, n + 1), k)
            lines.append(f"{k} " + " ".join(map(str, picks)))
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_86",
        title="Emergency Relay Cut",
        rating=2800,
        tags=["trees", "virtual tree", "dp", "lca", "divide and conquer"],
        notes="Weighted generalization of CF 613D: build a Virtual Tree per query with binary-lifted path-minimum edge capacities, then run 2-state tree DP (dp0: 0 terminals reach u, dp1: <=1 terminal reaches u) in O((N + sum K) log N).",
        story_and_spec="""A communication backbone connects N relay stations numbered 1 to N via N - 1 bidirectional fiber links forming a tree. Link i connects U_i and V_i and costs W_i credits to sever.

You are given Q independent containment scenarios. Scenario j specifies K_j compromised stations v_1, v_2, ..., v_{K_j} (all distinct): find the minimum total cost of fiber links to sever so that NO TWO compromised stations remain in the same connected component.

Input
The first line contains N and Q.
Each of the next N - 1 lines contains three integers U_i, V_i, and W_i.
Each of the next Q lines begins with K_j followed by K_j distinct station indices v_1 ... v_{K_j}.

Output
For each scenario print the minimum severing cost on its own line.

Constraints
2 <= N <= 150000
1 <= Q <= 120000
2 <= K_j <= N, and K_1 + K_2 + ... + K_Q <= 250000
1 <= W_i <= 10^8""",
        samples=[
            "6 3\n1 2 10\n1 3 5\n2 4 3\n2 5 4\n3 6 8\n3 4 5 6\n2 4 5\n3 2 4 6\n",
        ],
        small_inputs=[
            "2 1\n1 2 42\n2 1 2\n",
            gen_86(300, 300, 1800, 8601),
            gen_86(1400, 1200, 6000, 8602, "caterpillar"),
        ],
        large_inputs=[
            gen_86(150000, 100000, 240000, 8611, "chain"),
            gen_86(150000, 110000, 250000, 8612, "caterpillar"),
            gen_86(150000, 120000, 250000, 8613, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long INF = (1LL << 60);

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

    int LOG = __lg(n) + 2;
    vector<vector<int>> up(LOG, vector<int>(n + 1, 1));
    vector<vector<long long>> mnw(LOG, vector<long long>(n + 1, INF));
    vector<int> dep(n + 1, 0), tin(n + 1, 0);
    int timer = 0;
    {
        vector<pair<int, int>> st = {{1, 0}};
        dep[1] = 1;
        while (!st.empty()) {
            auto& [u, idx] = st.back();
            if (idx == 0) tin[u] = ++timer;
            if (idx < (int)adj[u].size()) {
                auto [v, w] = adj[u][idx++];
                if (v == up[0][u] && u != 1) continue;
                if (dep[v]) continue;
                dep[v] = dep[u] + 1;
                up[0][v] = u;
                mnw[0][v] = w;
                for (int b = 1; b < LOG; b++) {
                    int mid = up[b - 1][v];
                    up[b][v] = up[b - 1][mid];
                    mnw[b][v] = min(mnw[b - 1][v], mnw[b - 1][mid]);
                }
                st.push_back({v, 0});
            } else {
                st.pop_back();
            }
        }
    }

    auto lca = [&](int u, int v) {
        if (dep[u] < dep[v]) swap(u, v);
        int d = dep[u] - dep[v];
        for (int b = 0; b < LOG; b++) if ((d >> b) & 1) u = up[b][u];
        if (u == v) return u;
        for (int b = LOG - 1; b >= 0; b--)
            if (up[b][u] != up[b][v]) { u = up[b][u]; v = up[b][v]; }
        return up[0][u];
    };
    auto path_min = [&](int anc, int u) -> long long {
        long long res = INF;
        int d = dep[u] - dep[anc];
        for (int b = 0; b < LOG; b++) if ((d >> b) & 1) {
            res = min(res, mnw[b][u]);
            u = up[b][u];
        }
        return res;
    };

    vector<char> mark(n + 1, 0);
    vector<vector<pair<int, long long>>> vt(n + 1);
    vector<long long> dp0(n + 1, 0), dp1(n + 1, 0);
    vector<int> nodes, all_vt, stk;

    while (q--) {
        int k; cin >> k;
        nodes.resize(k);
        for (int i = 0; i < k; i++) {
            cin >> nodes[i];
            mark[nodes[i]] = 1;
        }
        sort(nodes.begin(), nodes.end(), [&](int a, int b) { return tin[a] < tin[b]; });

        all_vt.clear();
        stk.clear();
        auto push_vt = [&](int x) { all_vt.push_back(x); vt[x].clear(); };

        push_vt(nodes[0]);
        stk.push_back(nodes[0]);
        for (int i = 1; i < k; i++) {
            int u = nodes[i];
            int w = lca(u, stk.back());
            if (w != stk.back()) {
                while (stk.size() >= 2 && dep[stk[stk.size() - 2]] >= dep[w]) {
                    int child = stk.back(); stk.pop_back();
                    int par = stk.back();
                    vt[par].push_back({child, path_min(par, child)});
                }
                if (stk.back() != w) {
                    int child = stk.back(); stk.pop_back();
                    push_vt(w);
                    vt[w].push_back({child, path_min(w, child)});
                    stk.push_back(w);
                }
            }
            push_vt(u);
            stk.push_back(u);
        }
        while (stk.size() >= 2) {
            int child = stk.back(); stk.pop_back();
            int par = stk.back();
            vt[par].push_back({child, path_min(par, child)});
        }
        int root = stk[0];

        // Iterative post-order over virtual tree rooted at `root`
        vector<int> vord = {root};
        for (size_t i = 0; i < vord.size(); i++) {
            for (auto [c, _] : vt[vord[i]]) vord.push_back(c);
        }
        for (int i = (int)vord.size() - 1; i >= 0; i--) {
            int u = vord[i];
            long long sumA = 0, best_delta = 0;
            for (auto [c, mw] : vt[u]) {
                long long Ac = min(dp0[c], dp1[c] + mw);
                sumA += Ac;
                best_delta = min(best_delta, dp1[c] - Ac);
            }
            if (mark[u]) {
                dp0[u] = INF;
                dp1[u] = sumA;
            } else {
                dp0[u] = sumA;
                dp1[u] = sumA + best_delta;
            }
        }

        cout << dp1[root] << '\n';
        for (int x : nodes) mark[x] = 0;
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long INF = (1LL << 60);

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
    vector<char> mark(n + 1, 0);
    vector<long long> dp0(n + 1), dp1(n + 1);
    while (q--) {
        int k; cin >> k;
        vector<int> ks(k);
        for (int i = 0; i < k; i++) { cin >> ks[i]; mark[ks[i]] = 1; }
        for (int i = n - 1; i >= 0; i--) {
            int u = ord[i];
            long long sumA = 0, best_d = 0;
            for (auto [v, w] : adj[u]) if (v != par[u]) {
                long long Av = min(dp0[v], dp1[v] + w);
                sumA += Av;
                best_d = min(best_d, dp1[v] - Av);
            }
            if (mark[u]) {
                dp0[u] = INF;
                dp1[u] = sumA;
            } else {
                dp0[u] = sumA;
                dp1[u] = sumA + best_d;
            }
        }
        cout << dp1[1] << '\n';
        for (int x : ks) mark[x] = 0;
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_87: SUPPLY CHOKEPOINTS (2800) - Lengauer-Tarjan Dominator Tree
    # ------------------------------------------------------------------
    def gen_87(n: int, m: int, seed: int, long_chain: bool = False) -> str:
        r = random.Random(seed)
        edges = []
        if long_chain:
            # Spine 1 -> 2 -> ... -> n*3/4 with local cycles, diamonds, and back-edges
            sp = max(2, 3 * n // 4)
            for i in range(1, sp):
                edges.append((i, i + 1))
            for i in range(sp + 1, n + 1):
                edges.append((r.randint(1, sp), i))
            while len(edges) < m:
                u = r.randint(1, n)
                if r.random() < 0.7:
                    v = max(1, min(n, u + r.randint(-8, 8)))
                else:
                    v = r.randint(1, n)
                if u != v:
                    edges.append((u, v))
        else:
            for i in range(2, min(n + 1, m + 1)):
                edges.append((r.randint(1, i - 1), i))
            while len(edges) < m:
                u, v = r.randint(1, n), r.randint(1, n)
                if u != v:
                    edges.append((u, v))
        return f"{n} {len(edges)}\n" + "\n".join(f"{u} {v}" for u, v in edges) + "\n"

    specs.append(ProblemSpec(
        pid="synth_87",
        title="Supply Chokepoints",
        rating=2800,
        tags=["graphs", "dominator tree", "dfs and similar", "trees"],
        notes="Lengauer-Tarjan O((N + M) alpha(N)) Dominator Tree on general cyclic directed graphs; depth in the dominator tree gives the exact count of dominators for every reachable vertex.",
        story_and_spec="""A continental supply network consists of N depots numbered 1 to N and M one-way transport routes (which MAY contain directed cycles). Central command is located at depot 1.

For a depot v reachable from depot 1, a depot u (1 <= u <= N) is called a DOMINATOR of v if EVERY directed walk from depot 1 to depot v must visit u at least once (in particular, both depot 1 and depot v itself dominate v).

For every depot v = 1, 2, ..., N, compute how many dominators v has (or print 0 if depot v cannot be reached from depot 1).

Input
The first line contains N and M.
Each of the next M lines contains two integers U_i and V_i representing a directed route from U_i to V_i.

Output
Print N space-separated integers: the number of dominators of depots 1, 2, ..., N.

Constraints
1 <= N <= 150000
1 <= M <= 250000
1 <= U_i, V_i <= N, U_i != V_i""",
        samples=[
            "6 7\n1 2\n1 3\n2 4\n3 4\n4 2\n4 5\n3 6\n",
            "4 2\n1 2\n3 4\n",
        ],
        small_inputs=[
            "2 2\n1 2\n2 1\n",
            gen_87(250, 600, 8701, long_chain=True),
            gen_87(900, 2200, 8702, long_chain=False),
        ],
        large_inputs=[
            gen_87(150000, 250000, 8711, long_chain=True),
            gen_87(150000, 250000, 8712, long_chain=False),
            gen_87(140000, 240000, 8713, long_chain=True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
    }

    // Iterative DFS preorder from 1
    vector<int> dfn(n + 1, 0), rev_dfn(n + 2, 0), par(n + 2, 0);
    int cnt = 0;
    {
        struct Frame { int u, idx; };
        vector<Frame> st = {{1, 0}};
        while (!st.empty()) {
            auto& f = st.back();
            int u = f.u;
            if (f.idx == 0) {
                dfn[u] = ++cnt;
                rev_dfn[cnt] = u;
            }
            bool pushed = false;
            while (f.idx < (int)g[u].size()) {
                int v = g[u][f.idx++];
                if (!dfn[v]) {
                    par[cnt + 1] = dfn[u]; // assigned right on entry of v!
                    st.push_back({v, 0});
                    pushed = true;
                    break;
                }
            }
            if (!pushed) st.pop_back();
        }
    }

    // Build reverse graph in DFS-number coordinates 1..cnt
    vector<vector<int>> rg(cnt + 2), bucket(cnt + 2);
    for (int u = 1; u <= n; u++) if (dfn[u]) {
        int du = dfn[u];
        for (int v : g[u]) if (dfn[v]) {
            rg[dfn[v]].push_back(du);
        }
    }

    vector<int> sdom(cnt + 2), idom(cnt + 2, 0), dsu_p(cnt + 2), mn(cnt + 2);
    for (int i = 1; i <= cnt; i++) {
        sdom[i] = dsu_p[i] = mn[i] = i;
    }

    vector<int> chain;
    chain.reserve(cnt + 2);
    auto eval_dsu = [&](int x) -> int {
        if (dsu_p[x] == x) return x;
        chain.clear();
        int cur = x;
        while (dsu_p[cur] != cur) {
            chain.push_back(cur);
            cur = dsu_p[cur];
        }
        for (int idx = (int)chain.size() - 1; idx >= 0; idx--) {
            int v = chain[idx];
            int p = dsu_p[v];
            if (sdom[mn[p]] < sdom[mn[v]]) mn[v] = mn[p];
            dsu_p[v] = cur;
        }
        return mn[x];
    };

    for (int i = cnt; i >= 2; i--) {
        for (int v : rg[i]) {
            int u = eval_dsu(v);
            sdom[i] = min(sdom[i], sdom[u]);
        }
        bucket[sdom[i]].push_back(i);
        int p = par[i];
        dsu_p[i] = p;
        for (int w : bucket[p]) {
            int u = eval_dsu(w);
            idom[w] = (sdom[u] < sdom[w]) ? u : p;
        }
        bucket[p].clear();
    }

    vector<int> depth(cnt + 2, 0);
    depth[1] = 1;
    for (int i = 2; i <= cnt; i++) {
        if (idom[i] != sdom[i]) idom[i] = idom[idom[i]];
        depth[i] = depth[idom[i]] + 1;
    }

    for (int u = 1; u <= n; u++) {
        int ans = dfn[u] ? depth[dfn[u]] : 0;
        cout << ans << (u == n ? '\n' : ' ');
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
    vector<vector<int>> g(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u].push_back(v);
    }

    vector<char> reach0(n + 1, 0), vis(n + 1, 0);
    vector<int> qv = {1};
    reach0[1] = 1;
    for (size_t i = 0; i < qv.size(); i++) {
        for (int v : g[qv[i]]) if (!reach0[v]) {
            reach0[v] = 1;
            qv.push_back(v);
        }
    }

    vector<int> dom_cnt(n + 1, 0);
    for (int u = 1; u <= n; u++) if (reach0[u]) dom_cnt[u] = 1; // depot 1 dominates all reachable

    for (int ban = 2; ban <= n; ban++) {
        if (!reach0[ban]) continue;
        fill(vis.begin(), vis.end(), 0);
        vis[1] = 1;
        qv.assign(1, 1);
        for (size_t i = 0; i < qv.size(); i++) {
            int u = qv[i];
            for (int v : g[u]) if (v != ban && !vis[v]) {
                vis[v] = 1;
                qv.push_back(v);
            }
        }
        for (int v = 1; v <= n; v++) {
            if (reach0[v] && !vis[v]) dom_cnt[v]++;
        }
    }

    for (int u = 1; u <= n; u++) {
        cout << dom_cnt[u] << (u == n ? '\n' : ' ');
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_88: EPHEMERAL BRIDGES (2700) - Offline Dynamic Connectivity (Time SegTree + Rollback DSU)
    # ------------------------------------------------------------------
    def gen_88(n: int, q: int, seed: int, preload: bool = False) -> str:
        r = random.Random(seed)
        active: list[tuple[int, int]] = []
        active_set: set[tuple[int, int]] = set()
        lines = [f"{n} {q}"]
        init_adds = min(q // 3, n - 1) if preload else 0
        for i in range(q):
            if i == q - 1:
                lines.append("3")
            elif i < init_adds:
                u = i + 1
                v = r.randint(max(1, i - 4), i + 1)
                if u == v:
                    v = u + 1 if u < n else u - 1
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in active_set:
                    active_set.add((a, b))
                    active.append((a, b))
                    lines.append(f"1 {a} {b}")
                else:
                    lines.append("3")
            else:
                p = r.random()
                if p < 0.38 or (not active and p < 0.7):
                    for _ in range(10):
                        u, v = r.randint(1, n), r.randint(1, n)
                        if u != v:
                            a, b = (u, v) if u < v else (v, u)
                            if (a, b) not in active_set:
                                active_set.add((a, b))
                                active.append((a, b))
                                lines.append(f"1 {a} {b}")
                                break
                    else:
                        lines.append("3")
                elif p < 0.65 and active:
                    idx = r.randrange(len(active))
                    a, b = active[idx]
                    active[idx] = active[-1]
                    active.pop()
                    active_set.remove((a, b))
                    lines.append(f"2 {a} {b}")
                else:
                    lines.append("3")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_88",
        title="Ephemeral Bridges",
        rating=2700,
        tags=["graphs", "divide and conquer", "dsu", "segment tree", "offline queries"],
        notes="Offline Dynamic Connectivity: map each edge's active [t_add, t_del - 1] intervals into a Time Segment Tree and traverse with a Rollback DSU maintaining sum(|C|^2) in O(Q log Q log N).",
        story_and_spec="""An archipelago consists of N islands numbered 1 to N, initially with no bridges. Over Q time steps, three kinds of events occur:
- 1 U V: a bidirectional pontoon bridge is opened between islands U and V (guaranteed not currently open).
- 2 U V: the existing bridge between islands U and V is closed (guaranteed currently open).
- 3: report the current number of ORDERED PAIRS of islands (A, B) with 1 <= A, B <= N such that island B is reachable from island A via currently open bridges (including A = B).

Input
The first line contains N and Q.
Each of the next Q lines describes an event in one of the three formats above. At least one event is of type 3.

Output
For each event of type 3, print the number of mutually reachable ordered pairs on its own line.

Constraints
2 <= N <= 120000
1 <= Q <= 150000
1 <= U, V <= N, U != V""",
        samples=[
            "4 7\n3\n1 1 2\n1 3 4\n3\n1 2 3\n3\n2 1 2\n",
            "3 5\n1 1 2\n1 2 3\n1 1 3\n2 1 2\n3\n",
        ],
        small_inputs=[
            "2 3\n1 1 2\n3\n2 1 2\n",
            gen_88(250, 500, 8801),
            gen_88(1200, 1600, 8802, preload=True),
        ],
        large_inputs=[
            gen_88(120000, 150000, 8811, preload=True),
            gen_88(120000, 150000, 8812, preload=False),
            gen_88(100000, 150000, 8813, preload=True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;

    map<pair<int, int>, int> open_at;
    vector<char> is_q(q, 0);
    struct Span { int l, r, u, v; };
    vector<Span> spans;
    spans.reserve(q);

    for (int t = 0; t < q; t++) {
        int tp; cin >> tp;
        if (tp == 1) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            open_at[{u, v}] = t;
        } else if (tp == 2) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            auto it = open_at.find({u, v});
            spans.push_back({it->second, t - 1, u, v});
            open_at.erase(it);
        } else {
            is_q[t] = 1;
        }
    }
    for (auto& [e, start_t] : open_at) {
        spans.push_back({start_t, q - 1, e.first, e.second});
    }

    vector<vector<pair<int, int>>> seg(4 * q + 5);
    function<void(int, int, int, int, int, pair<int, int>)> add_span =
        [&](int node, int l, int r, int ql, int qr, pair<int, int> e) {
            if (ql <= l && r <= qr) {
                seg[node].push_back(e);
                return;
            }
            int mid = (l + r) >> 1;
            if (ql <= mid) add_span(node << 1, l, mid, ql, qr, e);
            if (qr > mid) add_span(node << 1 | 1, mid + 1, r, ql, qr, e);
        };
    for (const auto& s : spans) {
        if (s.l <= s.r) add_span(1, 0, q - 1, s.l, s.r, {s.u, s.v});
    }

    vector<int> p(n + 1), sz(n + 1, 1);
    iota(p.begin(), p.end(), 0);
    long long pairs = n;
    struct Undo { int b, a; long long prev_pairs; };
    vector<Undo> stk;
    stk.reserve(q * 2);

    auto find_root = [&](int x) {
        while (p[x] != x) x = p[x];
        return x;
    };
    auto unite = [&](int u, int v) {
        int a = find_root(u), b = find_root(v);
        if (a == b) return;
        if (sz[a] < sz[b]) swap(a, b);
        stk.push_back({b, a, pairs});
        pairs += 2LL * sz[a] * sz[b];
        p[b] = a;
        sz[a] += sz[b];
    };

    vector<long long> ans(q, -1);
    function<void(int, int, int)> dfs = [&](int node, int l, int r) {
        size_t snap = stk.size();
        for (auto [u, v] : seg[node]) unite(u, v);
        if (l == r) {
            if (is_q[l]) ans[l] = pairs;
        } else {
            int mid = (l + r) >> 1;
            dfs(node << 1, l, mid);
            dfs(node << 1 | 1, mid + 1, r);
        }
        while (stk.size() > snap) {
            auto un = stk.back(); stk.pop_back();
            sz[un.a] -= sz[un.b];
            p[un.b] = un.b;
            pairs = un.prev_pairs;
        }
    };

    dfs(1, 0, q - 1);
    for (int t = 0; t < q; t++) if (is_q[t]) cout << ans[t] << '\n';
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
    set<pair<int, int>> edges;
    vector<int> p(n + 1), sz(n + 1);
    auto findp = [&](auto& self, int x) -> int { return p[x] == x ? x : p[x] = self(self, p[x]); };

    while (q--) {
        int tp; cin >> tp;
        if (tp == 1) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            edges.insert({u, v});
        } else if (tp == 2) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            edges.erase({u, v});
        } else {
            iota(p.begin(), p.end(), 0);
            fill(sz.begin(), sz.end(), 1);
            long long pairs = n;
            for (auto [u, v] : edges) {
                int a = findp(findp, u), b = findp(findp, v);
                if (a != b) {
                    pairs += 2LL * sz[a] * sz[b];
                    p[b] = a;
                    sz[a] += sz[b];
                }
            }
            cout << pairs << '\n';
        }
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_89: IMPERIAL RESEARCH GRANTS (2600) - Interval Project Selection Closure (CF 115E)
    # ------------------------------------------------------------------
    def gen_89(n: int, m: int, seed: int) -> str:
        r = random.Random(seed)
        c = [r.randint(1, 10**8) for _ in range(n)]
        lines = [f"{n} {m}", " ".join(map(str, c))]
        for _ in range(m):
            l, rr = r.randint(1, n), r.randint(1, n)
            if l > rr:
                l, rr = rr, l
            p = r.randint(1, 2 * 10**8)
            lines.append(f"{l} {rr} {p}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_89",
        title="Imperial Research Grants",
        rating=2600,
        tags=["dp", "segment tree", "lazy propagation", "flows", "min-cut"],
        notes="Interval Project Selection Closure (CF 115E): sweep right endpoint r = 1..N with a lazy range-add range-max Segment Tree tracking net profit of contiguous repaired block [j..r] in O((N + M) log N).",
        story_and_spec="""Along a trade corridor lie N sequential highway sections numbered 1 to N. Rehabilitating section i costs C_i gold.

There are M merchant guild charters: charter j pays a bonus of P_j gold IF AND ONLY IF EVERY highway section k satisfying L_j <= k <= R_j has been rehabilitated.

Choose a subset of highway sections to rehabilitate (possibly empty) maximizing your NET PROFIT:
(total bonuses of all satisfied charters) - (total rehabilitation cost of chosen sections).

Input
The first line contains N and M.
The second line contains C_1 ... C_N.
Each of the next M lines contains L_j, R_j, and P_j.

Output
Print a single integer: the maximum achievable net profit.

Constraints
1 <= N, M <= 160000
0 <= C_i, P_j <= 10^9
1 <= L_j <= R_j <= N""",
        samples=[
            "5 4\n4 3 5 2 6\n1 2 8\n2 4 9\n4 5 7\n1 5 3\n",
            "3 2\n10 10 10\n1 2 5\n2 3 6\n",
        ],
        small_inputs=[
            "1 1\n5\n1 1 9\n",
            gen_89(250, 400, 8901),
            gen_89(1400, 2000, 8902),
        ],
        large_inputs=[
            gen_89(160000, 160000, 8911),
            gen_89(160000, 160000, 8912),
            gen_89(160000, 160000, 8913),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long NEG_INF = -(1LL << 60);

struct SegTree {
    int n;
    vector<long long> mx, lz;
    SegTree(int n) : n(n), mx(4 * n + 5, NEG_INF), lz(4 * n + 5, 0) {}

    void apply(int u, long long v) {
        mx[u] += v;
        lz[u] += v;
    }
    void push(int u) {
        if (lz[u] != 0) {
            apply(u << 1, lz[u]);
            apply(u << 1 | 1, lz[u]);
            lz[u] = 0;
        }
    }
    void set_point(int u, int l, int r, int pos, long long val) {
        if (l == r) { mx[u] = val; lz[u] = 0; return; }
        push(u);
        int mid = (l + r) >> 1;
        if (pos <= mid) set_point(u << 1, l, mid, pos, val);
        else set_point(u << 1 | 1, mid + 1, r, pos, val);
        mx[u] = max(mx[u << 1], mx[u << 1 | 1]);
    }
    void add_range(int u, int l, int r, int ql, int qr, long long val) {
        if (ql <= l && r <= qr) { apply(u, val); return; }
        push(u);
        int mid = (l + r) >> 1;
        if (ql <= mid) add_range(u << 1, l, mid, ql, qr, val);
        if (qr > mid) add_range(u << 1 | 1, mid + 1, r, ql, qr, val);
        mx[u] = max(mx[u << 1], mx[u << 1 | 1]);
    }
    long long query_max(int u, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return mx[u];
        push(u);
        int mid = (l + r) >> 1;
        long long res = NEG_INF;
        if (ql <= mid) res = max(res, query_max(u << 1, l, mid, ql, qr));
        if (qr > mid) res = max(res, query_max(u << 1 | 1, mid + 1, r, ql, qr));
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<long long> c(n + 1);
    for (int i = 1; i <= n; i++) cin >> c[i];
    vector<vector<pair<int, long long>>> by_r(n + 1);
    for (int i = 0; i < m; i++) {
        int l, r; long long p;
        cin >> l >> r >> p;
        by_r[r].push_back({l, p});
    }

    SegTree st(n);
    vector<long long> f(n + 1, 0);
    for (int r = 1; r <= n; r++) {
        long long base = (r >= 2 ? f[r - 2] : 0LL);
        st.set_point(1, 1, n, r, base);
        st.add_range(1, 1, n, 1, r, -c[r]);
        for (auto [l, p] : by_r[r]) {
            st.add_range(1, 1, n, 1, l, p);
        }
        f[r] = max(f[r - 1], st.query_max(1, 1, n, 1, r));
    }
    cout << f[n] << '\n';
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
    vector<long long> c(n + 1);
    for (int i = 1; i <= n; i++) cin >> c[i];
    vector<vector<pair<int, long long>>> by_r(n + 1);
    for (int i = 0; i < m; i++) {
        int l, r; long long p;
        cin >> l >> r >> p;
        by_r[r].push_back({l, p});
    }

    vector<long long> f(n + 1, 0), gain_at_l(n + 1, 0);
    for (int r = 1; r <= n; r++) {
        for (auto [l, p] : by_r[r]) gain_at_l[l] += p;
        long long best = f[r - 1];
        long long cur_gain = 0, cur_cost = 0;
        for (int j = r; j >= 1; j--) {
            cur_gain += gain_at_l[j];
            cur_cost += c[j];
            long long prev = (j >= 2 ? f[j - 2] : 0LL);
            best = max(best, prev + cur_gain - cur_cost);
        }
        f[r] = best;
    }
    cout << f[n] << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_90: BIPARTITE DOCKYARD (2800) - Weighted Prefix-Suffix Bipartite Flow (ARC076F)
    # ------------------------------------------------------------------
    def gen_90(n: int, m: int, seed: int) -> str:
        r = random.Random(seed)
        caps = [r.randint(0, 10**6) for _ in range(m)]
        lines = [f"{n} {m}", " ".join(map(str, caps))]
        for _ in range(n):
            # Ensure [1..L] U [R..M] is non-empty: either L >= 1 or R <= M
            while True:
                l = r.randint(0, m)
                rr = r.randint(l + 1, m + 1)
                if l >= 1 or rr <= m:
                    break
            cnt = r.randint(1, 2 * 10**6)
            lines.append(f"{l} {rr} {cnt}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_90",
        title="Bipartite Dockyard",
        rating=2800,
        tags=["flows", "min-cut", "hall theorem", "segment tree", "greedy"],
        notes="Weighted generalization of ARC076F: by s-t Min-Cut / Hall's Theorem, uncut piers form a single middle interval [a+1..b-1]; sweep a = 0..M-1 with a lazy range-add range-min Segment Tree over b in O((N + M) log M). Cross-checked against 64-bit Dinic Max-Flow on small!",
        story_and_spec="""A harbor has M piers numbered 1 to M in a row from west to east. Pier p can accommodate at most Cap_p vessels simultaneously.

N merchant fleets arrive seeking berths. Fleet i consists of C_i identical vessels, and because of draft restrictions in the central channel, each vessel of fleet i may berth at ANY pier p satisfying:
1 <= p <= L_i   OR   R_i <= p <= M
(note that L_i = 0 means no western piers are usable, and R_i = M + 1 means no eastern piers are usable; different vessels of the same fleet may use different piers, and you may leave some vessels unberthed).

Compute the MAXIMUM total number of vessels that can be berthed simultaneously across all M piers.

Input
The first line contains N and M.
The second line contains Cap_1 ... Cap_M.
Each of the next N lines contains three integers L_i, R_i, and C_i.

Output
Print a single integer: the maximum number of vessels that can be berthed.

Constraints
1 <= N, M <= 180000
0 <= Cap_p <= 10^8
0 <= L_i < R_i <= M + 1, and at least one of L_i >= 1 or R_i <= M holds
1 <= C_i <= 10^8""",
        samples=[
            "4 4\n2 1 2 3\n1 4 3\n2 5 2\n0 3 4\n1 3 2\n",
            "3 3\n5 5 5\n1 4 4\n2 4 4\n0 2 4\n",
        ],
        small_inputs=[
            "2 1\n10\n1 2 6\n0 1 7\n",
            gen_90(120, 120, 9001),
            gen_90(220, 200, 9002),
        ],
        large_inputs=[
            gen_90(180000, 180000, 9011),
            gen_90(180000, 180000, 9012),
            gen_90(180000, 180000, 9013),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long INF = (1LL << 61);

struct SegTree {
    int L0, R0;
    vector<long long> mn, lz;
    SegTree(int l0, int r0) : L0(l0), R0(r0) {
        int sz = max(1, r0 - l0 + 1);
        mn.assign(4 * sz + 5, INF);
        lz.assign(4 * sz + 5, 0);
    }
    void apply(int u, long long v) { mn[u] += v; lz[u] += v; }
    void push(int u) {
        if (lz[u] != 0) {
            apply(u << 1, lz[u]);
            apply(u << 1 | 1, lz[u]);
            lz[u] = 0;
        }
    }
    void build(int u, int l, int r, const vector<long long>& init) {
        if (l == r) { mn[u] = init[l]; return; }
        int mid = (l + r) >> 1;
        build(u << 1, l, mid, init);
        build(u << 1 | 1, mid + 1, r, init);
        mn[u] = min(mn[u << 1], mn[u << 1 | 1]);
    }
    void add(int u, int l, int r, int ql, int qr, long long val) {
        if (ql > qr) return;
        if (ql <= l && r <= qr) { apply(u, val); return; }
        push(u);
        int mid = (l + r) >> 1;
        if (ql <= mid) add(u << 1, l, mid, ql, qr, val);
        if (qr > mid) add(u << 1 | 1, mid + 1, r, ql, qr, val);
        mn[u] = min(mn[u << 1], mn[u << 1 | 1]);
    }
    long long query(int u, int l, int r, int ql, int qr) {
        if (ql > qr) return INF;
        if (ql <= l && r <= qr) return mn[u];
        push(u);
        int mid = (l + r) >> 1;
        long long res = INF;
        if (ql <= mid) res = min(res, query(u << 1, l, mid, ql, qr));
        if (qr > mid) res = min(res, query(u << 1 | 1, mid + 1, r, ql, qr));
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<long long> cap(m + 2, 0), pref(m + 2, 0), suff(m + 3, 0);
    for (int i = 1; i <= m; i++) {
        cin >> cap[i];
        pref[i] = pref[i - 1] + cap[i];
    }
    for (int i = m; i >= 1; i--) suff[i] = suff[i + 1] + cap[i];

    vector<vector<pair<int, long long>>> by_l(m + 1);
    long long tot_c = 0;
    for (int i = 0; i < n; i++) {
        int l, r; long long c;
        cin >> l >> r >> c;
        tot_c += c;
        if (l < m) by_l[l].push_back({r, c});
    }

    // Cut Case 1: all piers cut (capacity pref[m]), or all fleets cut (tot_c)
    long long ans = min(tot_c, pref[m]);

    // Cut Case 2: uncut piers are [a+1 .. b-1] for 0 <= a <= m-1 and a+2 <= b <= m+1
    SegTree st(2, m + 1);
    st.build(1, 2, m + 1, suff);

    for (int a = 0; a <= m - 1; a++) {
        for (auto [r, c] : by_l[a]) {
            if (r >= 2) st.add(1, 2, m + 1, 2, min(r, m + 1), -c);
        }
        long long best_b = st.query(1, 2, m + 1, a + 2, m + 1);
        ans = min(ans, pref[a] + tot_c + best_b);
    }

    cout << ans << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct Dinic {
    struct E { int to, rev; long long cap; };
    int n;
    vector<vector<E>> g;
    vector<int> lvl, ptr;
    Dinic(int n) : n(n), g(n), lvl(n), ptr(n) {}
    void add_edge(int u, int v, long long c) {
        g[u].push_back({v, (int)g[v].size(), c});
        g[v].push_back({u, (int)g[u].size() - 1, 0});
    }
    long long dfs(int u, int t, long long pushed) {
        if (pushed == 0 || u == t) return pushed;
        for (int& cid = ptr[u]; cid < (int)g[u].size(); cid++) {
            auto& e = g[u][cid];
            if (lvl[e.to] != lvl[u] + 1 || e.cap == 0) continue;
            long long tr = dfs(e.to, t, min(pushed, e.cap));
            if (tr == 0) continue;
            e.cap -= tr;
            g[e.to][e.rev].cap += tr;
            return tr;
        }
        return 0;
    }
    long long flow(int s, int t) {
        long long f = 0;
        while (true) {
            fill(lvl.begin(), lvl.end(), -1);
            lvl[s] = 0;
            queue<int> q;
            q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (auto& e : g[u]) if (e.cap > 0 && lvl[e.to] == -1) {
                    lvl[e.to] = lvl[u] + 1;
                    q.push(e.to);
                }
            }
            if (lvl[t] == -1) break;
            fill(ptr.begin(), ptr.end(), 0);
            while (long long tr = dfs(s, t, (1LL << 60))) f += tr;
        }
        return f;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<long long> cap(m + 1);
    for (int i = 1; i <= m; i++) cin >> cap[i];
    struct Fleet { int l, r; long long c; };
    vector<Fleet> fl(n);
    for (int i = 0; i < n; i++) cin >> fl[i].l >> fl[i].r >> fl[i].c;

    if (n <= 400 && m <= 400) {
        // Exact Dinic Max-Flow on small inputs!
        int S = 0, T = n + m + 1;
        Dinic din(T + 1);
        const long long INF = (1LL << 60);
        for (int i = 0; i < n; i++) {
            din.add_edge(S, 1 + i, fl[i].c);
            for (int p = 1; p <= fl[i].l; p++) din.add_edge(1 + i, 1 + n + p - 1, INF);
            for (int p = max(fl[i].l + 1, fl[i].r); p <= m; p++) din.add_edge(1 + i, 1 + n + p - 1, INF);
        }
        for (int p = 1; p <= m; p++) din.add_edge(1 + n + p - 1, T, cap[p]);
        cout << din.flow(S, T) << '\n';
        return 0;
    }

    // O(M * N + M^2) memory-safe evaluation for large[0] TLE verification
    vector<long long> pref(m + 2, 0), suff(m + 3, 0);
    for (int i = 1; i <= m; i++) pref[i] = pref[i - 1] + cap[i];
    for (int i = m; i >= 1; i--) suff[i] = suff[i + 1] + cap[i];
    long long tot_c = 0;
    for (auto& f : fl) tot_c += f.c;
    long long ans = min(tot_c, pref[m]);
    for (int a = 0; a <= m - 1; a++) {
        for (int b = a + 2; b <= m + 1; b++) {
            long long saved = 0;
            for (auto& f : fl) if (f.l <= a && f.r >= b) saved += f.c;
            ans = min(ans, pref[a] + suff[b] + tot_c - saved);
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_18())
