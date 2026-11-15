"""Batch 22: synth_106 .. synth_110 (Grandmaster Wave VI: CF 2800-2900)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from builder import ProblemSpec, build_and_verify_batch


# ---------------------------------------------------------------------------
# synth_106 — Tree Path Min-Cut / Patrol Checkpoints (HLD + SegTree Dinic, CF 786E)
# ---------------------------------------------------------------------------
FAST_106 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

struct Dinic {
    struct Edge {
        int to, rev;
        long long cap;
    };
    int n;
    vector<vector<Edge>> g;
    vector<int> lvl, ptr;

    explicit Dinic(int n_) : n(n_), g(n_), lvl(n_), ptr(n_) {}

    void add_edge(int u, int v, long long cap) {
        Edge a{v, (int)g[v].size(), cap};
        Edge b{u, (int)g[u].size(), 0};
        g[u].push_back(a);
        g[v].push_back(b);
    }

    bool bfs(int s, int t) {
        fill(lvl.begin(), lvl.end(), -1);
        queue<int> q;
        lvl[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (const auto& e : g[u]) {
                if (e.cap > 0 && lvl[e.to] == -1) {
                    lvl[e.to] = lvl[u] + 1;
                    q.push(e.to);
                }
            }
        }
        return lvl[t] != -1;
    }

    long long dfs(int u, int t, long long pushed) {
        if (pushed == 0 || u == t) return pushed;
        for (int& cid = ptr[u]; cid < (int)g[u].size(); cid++) {
            auto& e = g[u][cid];
            if (e.cap > 0 && lvl[e.to] == lvl[u] + 1) {
                long long tr = dfs(e.to, t, min(pushed, e.cap));
                if (tr > 0) {
                    e.cap -= tr;
                    g[e.to][e.rev].cap += tr;
                    return tr;
                }
            }
        }
        return 0;
    }

    long long max_flow(int s, int t) {
        long long flow = 0;
        while (bfs(s, t)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (long long tr = dfs(s, t, INF)) flow += tr;
        }
        return flow;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<tuple<int, long long, int>>> adj(n + 1);
    for (int i = 1; i <= n - 1; i++) {
        int u, v;
        long long c;
        cin >> u >> v >> c;
        adj[u].push_back({v, c, i});
        adj[v].push_back({u, c, i});
    }

    vector<int> par(n + 1, 0), dep(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0);
    vector<long long> up_cost(n + 1, 0);
    vector<int> ord;
    ord.reserve(n);
    ord.push_back(1);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, c, id] : adj[u]) {
            if (v != par[u]) {
                par[v] = u;
                dep[v] = dep[u] + 1;
                up_cost[v] = c;
                ord.push_back(v);
            }
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        int best = 0;
        for (auto [v, c, id] : adj[u]) {
            if (v != par[u]) {
                sz[u] += sz[v];
                if (sz[v] > best) {
                    best = sz[v];
                    heavy[u] = v;
                }
            }
        }
    }

    vector<int> head(n + 1, 0), tin(n + 1, 0), inv_tin(n + 1, 0);
    int timer = 0;
    vector<pair<int, int>> stk;
    stk.push_back({1, 1});
    while (!stk.empty()) {
        auto [u, h] = stk.back();
        stk.pop_back();
        for (int x = u; x != 0; x = heavy[x]) {
            head[x] = h;
            tin[x] = ++timer;
            inv_tin[timer] = x;
            for (auto [v, c, id] : adj[x]) {
                if (v != par[x] && v != heavy[x]) {
                    stk.push_back({v, v});
                }
            }
        }
    }

    // Nodes in flow network:
    // 0: source, 1: sink
    // [2 .. 2 + 4*n]: segtree over HLD positions [2..n] (tin[1]=1 has no parent edge)
    // [2 + 4*n + 1 .. 2 + 4*n + m]: courier nodes
    int seg_base = 2;
    int cour_base = seg_base + 4 * n + 2;
    Dinic dinic(cour_base + m + 2);
    int S = 0, T = 1;

    if (n >= 2) {
        auto build_seg = [&](auto& self, int u, int l, int r) -> void {
            if (l == r) {
                int vtx = inv_tin[l];
                dinic.add_edge(seg_base + u, T, up_cost[vtx]);
                return;
            }
            int mid = (l + r) >> 1;
            self(self, u << 1, l, mid);
            self(self, u << 1 | 1, mid + 1, r);
            dinic.add_edge(seg_base + u, seg_base + (u << 1), INF);
            dinic.add_edge(seg_base + u, seg_base + (u << 1 | 1), INF);
        };
        build_seg(build_seg, 1, 2, n);
    }

    auto add_range = [&](auto& self, int cnode, int u, int l, int r, int ql, int qr) -> void {
        if (ql > qr) return;
        if (ql <= l && r <= qr) {
            dinic.add_edge(cnode, seg_base + u, INF);
            return;
        }
        int mid = (l + r) >> 1;
        if (ql <= mid) self(self, cnode, u << 1, l, mid, ql, qr);
        if (qr > mid) self(self, cnode, u << 1 | 1, mid + 1, r, ql, qr);
    };

    for (int j = 0; j < m; j++) {
        int u, v;
        long long p;
        cin >> u >> v >> p;
        if (u == v) continue;
        int cnode = cour_base + j;
        dinic.add_edge(S, cnode, p);
        while (head[u] != head[v]) {
            if (dep[head[u]] < dep[head[v]]) swap(u, v);
            add_range(add_range, cnode, 1, 2, n, max(2, tin[head[u]]), tin[u]);
            u = par[head[u]];
        }
        if (u != v) {
            if (dep[u] > dep[v]) swap(u, v);
            add_range(add_range, cnode, 1, 2, n, tin[u] + 1, tin[v]);
        }
    }

    cout << dinic.max_flow(S, T) << '\n';
    return 0;
}
"""

BRUTE_106 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

// Memory-safe O(N + M + augmentations) implicit Dinic/Edmonds-Karp:
// Forward edges courier j -> tree edge e have INF capacity so they are walked
// directly along the tree path without allocating O(N * M) edges!
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<vector<pair<int, int>>> adj(n + 1);
    vector<long long> ecap(n, 0);
    for (int i = 1; i <= n - 1; i++) {
        int u, v;
        long long c;
        cin >> u >> v >> c;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
        ecap[i] = c;
    }

    vector<int> par(n + 1, 0), par_edge(n + 1, 0), dep(n + 1, 0), q_bfs(n + 1);
    int qh = 0, qt = 0;
    q_bfs[qt++] = 1;
    while (qh < qt) {
        int u = q_bfs[qh++];
        for (auto [v, id] : adj[u]) {
            if (v != par[u]) {
                par[v] = u;
                par_edge[v] = id;
                dep[v] = dep[u] + 1;
                q_bfs[qt++] = v;
            }
        }
    }

    vector<int> cu(m), cv(m);
    vector<long long> scap(m);
    for (int j = 0; j < m; j++) {
        cin >> cu[j] >> cv[j] >> scap[j];
        if (cu[j] == cv[j]) scap[j] = 0;
    }

    // Sparse positive flow f[e][j] > 0 stored per tree edge e
    vector<unordered_map<int, long long>> rev_flow(n);

    // Node numbering for BFS:
    // 0: S, 1..m: couriers, m+1 .. m+n-1: tree edges 1..n-1, m+n: T
    int S = 0, T = m + n;
    int V = T + 1;
    vector<int> prev_node(V), que(V);
    vector<long long> bneck(V);

    long long total_flow = 0;
    while (true) {
        fill(prev_node.begin(), prev_node.end(), -1);
        int head = 0, tail = 0;
        prev_node[S] = S;
        bneck[S] = INF;
        que[tail++] = S;

        while (head < tail && prev_node[T] == -1) {
            int x = que[head++];
            if (x == S) {
                for (int j = 0; j < m; j++) {
                    if (scap[j] > 0 && prev_node[1 + j] == -1) {
                        prev_node[1 + j] = S;
                        bneck[1 + j] = scap[j];
                        que[tail++] = 1 + j;
                    }
                }
            } else if (x >= 1 && x <= m) {
                int j = x - 1;
                int a = cu[j], b = cv[j];
                while (a != b) {
                    if (dep[a] < dep[b]) swap(a, b);
                    int eid = par_edge[a];
                    int enode = m + eid;
                    if (prev_node[enode] == -1) {
                        prev_node[enode] = x;
                        bneck[enode] = bneck[x];
                        que[tail++] = enode;
                        if (ecap[eid] > 0 && prev_node[T] == -1) {
                            prev_node[T] = enode;
                            bneck[T] = min(bneck[enode], ecap[eid]);
                            break;
                        }
                    }
                    a = par[a];
                }
            } else if (x > m && x < T) {
                int eid = x - m;
                if (ecap[eid] > 0 && prev_node[T] == -1) {
                    prev_node[T] = x;
                    bneck[T] = min(bneck[x], ecap[eid]);
                    break;
                }
                for (const auto& [j, fl] : rev_flow[eid]) {
                    if (fl > 0 && prev_node[1 + j] == -1) {
                        prev_node[1 + j] = x;
                        bneck[1 + j] = min(bneck[x], fl);
                        que[tail++] = 1 + j;
                    }
                }
            }
        }

        if (prev_node[T] == -1) break;
        long long push = bneck[T];
        total_flow += push;

        int cur = T;
        while (cur != S) {
            int p = prev_node[cur];
            if (cur == T) {
                ecap[p - m] -= push;
            } else if (p == S) {
                scap[cur - 1] -= push;
            } else if (p >= 1 && p <= m) {
                int j = p - 1, eid = cur - m;
                rev_flow[eid][j] += push;
            } else {
                int eid = p - m, j = cur - 1;
                auto it = rev_flow[eid].find(j);
                it->second -= push;
                if (it->second == 0) rev_flow[eid].erase(it);
            }
            cur = p;
        }
    }

    cout << total_flow << '\n';
    return 0;
}
"""


def gen_106(n: int, m: int, seed: int, chain: bool = False) -> str:
    r = random.Random(seed)
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    lines = [f"{n} {m}"]
    for i in range(1, n):
        p = perm[i - 1] if chain else perm[r.randint(max(0, i - 6), i - 1)]
        c = r.randint(1, 10**6)
        lines.append(f"{p} {perm[i]} {c}")
    for _ in range(m):
        if chain and r.random() < 0.85:
            u = perm[r.randint(0, max(0, n // 6))]
            v = perm[r.randint(max(0, n - 1 - n // 6), n - 1)]
        else:
            u = r.randint(1, n)
            v = r.randint(1, n)
        p = r.randint(1, 10**6)
        lines.append(f"{u} {v} {p}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_107 — Minkowski Sum Support Queries (Minkowski Sum + Hull Binary Search)
# ---------------------------------------------------------------------------
FAST_107 = r"""
#include <bits/stdc++.h>
using namespace std;

struct Pt {
    long long x, y;
    Pt operator+(const Pt& o) const { return {x + o.x, y + o.y}; }
    Pt operator-(const Pt& o) const { return {x - o.x, y - o.y}; }
};

static inline __int128 cross(const Pt& a, const Pt& b) {
    return (__int128)a.x * b.y - (__int128)a.y * b.x;
}

static inline __int128 cross3(const Pt& o, const Pt& a, const Pt& b) {
    return cross(a - o, b - o);
}

static inline int half_plane(const Pt& v) {
    // Angle in [0, pi) -> 0, [pi, 2pi) -> 1
    return (v.y > 0 || (v.y == 0 && v.x > 0)) ? 0 : 1;
}

vector<Pt> convex_hull_ccw(vector<Pt>& p) {
    sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });
    p.erase(unique(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        return a.x == b.x && a.y == b.y;
    }), p.end());
    int n = (int)p.size();
    if (n <= 2) return p;
    vector<Pt> h;
    h.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        while (h.size() >= 2 && cross3(h[h.size() - 2], h.back(), p[i]) <= 0) h.pop_back();
        h.push_back(p[i]);
    }
    size_t t = h.size() + 1;
    for (int i = n - 2; i >= 0; i--) {
        while (h.size() >= t && cross3(h[h.size() - 2], h.back(), p[i]) <= 0) h.pop_back();
        h.push_back(p[i]);
    }
    h.pop_back();
    return h;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    Pt start{0, 0};
    vector<Pt> edges;
    edges.reserve(300000);

    for (int i = 0; i < n; i++) {
        int m;
        cin >> m;
        vector<Pt> pts(m);
        for (int j = 0; j < m; j++) cin >> pts[j].x >> pts[j].y;
        vector<Pt> h = convex_hull_ccw(pts);
        int sz = (int)h.size();
        // Rotate so lowest-then-leftmost vertex is first
        int idx = 0;
        for (int j = 1; j < sz; j++) {
            if (h[j].y < h[idx].y || (h[j].y == h[idx].y && h[j].x < h[idx].x)) idx = j;
        }
        rotate(h.begin(), h.begin() + idx, h.end());
        start = start + h[0];
        if (sz == 2) {
            edges.push_back(h[1] - h[0]);
            edges.push_back(h[0] - h[1]);
        } else if (sz >= 3) {
            for (int j = 0; j < sz; j++) {
                edges.push_back(h[(j + 1) % sz] - h[j]);
            }
        }
    }

    sort(edges.begin(), edges.end(), [](const Pt& a, const Pt& b) {
        int ha = half_plane(a), hb = half_plane(b);
        if (ha != hb) return ha < hb;
        return cross(a, b) > 0;
    });

    vector<Pt> verts;
    verts.reserve(edges.size() + 1);
    Pt cur = start;
    verts.push_back(cur);
    for (const auto& e : edges) {
        cur = cur + e;
        verts.push_back(cur);
    }

    // Extract clean strict-x upper and lower chains for branch-free binary search
    sort(verts.begin(), verts.end(), [](const Pt& a, const Pt& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });

    vector<Pt> lower, upper;
    for (const auto& p : verts) {
        if (!lower.empty() && lower.back().x == p.x) continue; // keep min y for x
        while (lower.size() >= 2 && cross3(lower[lower.size() - 2], lower.back(), p) <= 0)
            lower.pop_back();
        lower.push_back(p);
    }
    for (int i = 0; i < (int)verts.size(); i++) {
        if (i + 1 < (int)verts.size() && verts[i + 1].x == verts[i].x) continue; // keep max y for x
        const auto& p = verts[i];
        while (upper.size() >= 2 && cross3(upper[upper.size() - 2], upper.back(), p) >= 0)
            upper.pop_back();
        upper.push_back(p);
    }

    auto max_on_chain = [](const vector<Pt>& ch, long long A, long long B) -> long long {
        int lo = 0, hi = (int)ch.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            long long dx = ch[mid + 1].x - ch[mid].x;
            long long dy = ch[mid + 1].y - ch[mid].y;
            if ((__int128)A * dx + (__int128)B * dy > 0) lo = mid + 1;
            else hi = mid;
        }
        return A * ch[lo].x + B * ch[lo].y;
    };

    while (q--) {
        long long A, B;
        cin >> A >> B;
        long long ans;
        if (B >= 0) ans = max_on_chain(upper, A, B);
        else ans = max_on_chain(lower, A, B);
        // Also check opposite chain endpoints in case B == 0
        ans = max({ans, A * lower.front().x + B * lower.front().y,
                        A * lower.back().x + B * lower.back().y});
        cout << ans << '\n';
    }
    return 0;
}
"""

BRUTE_107 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<long long, long long>>> sets(n);
    for (int i = 0; i < n; i++) {
        int m;
        cin >> m;
        sets[i].resize(m);
        for (int j = 0; j < m; j++) cin >> sets[i][j].first >> sets[i][j].second;
    }
    while (q--) {
        long long A, B;
        cin >> A >> B;
        long long total = 0;
        for (int i = 0; i < n; i++) {
            long long best = -(1LL << 62);
            for (auto [x, y] : sets[i]) {
                best = max(best, A * x + B * y);
            }
            total += best;
        }
        cout << total << '\n';
    }
    return 0;
}
"""


def gen_107(n: int, tot_pts: int, q: int, seed: int) -> str:
    r = random.Random(seed)
    import math
    lines = [f"{n} {q}"]
    rem = tot_pts
    for i in range(n):
        left_groups = n - 1 - i
        if left_groups == 0:
            m = rem
        else:
            avg = max(1, (rem - left_groups) // (left_groups + 1))
            m = r.randint(1, min(rem - left_groups, max(1, 2 * avg)))
        rem -= m
        lines.append(str(m))
        cx = r.randint(-500000, 500000)
        cy = r.randint(-500000, 500000)
        rad = r.randint(50, 400000)
        for _ in range(m):
            ang = r.random() * 2.0 * math.pi
            px = int(cx + rad * math.cos(ang)) + r.randint(-5, 5)
            py = int(cy + rad * math.sin(ang)) + r.randint(-5, 5)
            lines.append(f"{px} {py}")
    for _ in range(q):
        while True:
            a = r.randint(-10**6, 10**6)
            b = r.randint(-10**6, 10**6)
            if a != 0 or b != 0:
                break
        lines.append(f"{a} {b}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_108 — Dynamic Multiset LCM Pair Sum (Dirichlet / Mobius Inversion)
# ---------------------------------------------------------------------------
FAST_108 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;
static const int MAXV = 300000;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<int> a(n + 1);
    int vmax = 1;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        vmax = max(vmax, a[i]);
    }

    vector<int> inv(vmax + 1, 1), mu(vmax + 1, 0), lp(vmax + 1, 0), primes;
    mu[1] = 1;
    for (int i = 2; i <= vmax; i++) {
        inv[i] = (int)(1LL * (MOD - MOD / i) * inv[MOD % i] % MOD);
        if (!lp[i]) {
            lp[i] = i;
            primes.push_back(i);
            mu[i] = -1;
        }
        for (int p : primes) {
            if (p > lp[i] || 1LL * i * p > vmax) break;
            lp[i * p] = p;
            if (p == lp[i]) {
                mu[i * p] = 0;
                break;
            } else {
                mu[i * p] = -mu[i];
            }
        }
    }

    // f(n) = sum_{d | n} mu(n / d) * d^{-1} mod MOD
    vector<int> f(vmax + 1, 0);
    for (int d = 1; d <= vmax; d++) {
        int id = inv[d];
        for (int m = d, k = 1; m <= vmax; m += d, k++) {
            if (mu[k] == 1) {
                f[m] += id;
                if (f[m] >= MOD) f[m] -= MOD;
            } else if (mu[k] == -1) {
                f[m] -= id;
                if (f[m] < 0) f[m] += MOD;
            }
        }
    }

    // Precompute divisors only for values present in array a
    vector<char> present(vmax + 1, 0);
    for (int i = 1; i <= n; i++) present[a[i]] = 1;
    vector<vector<int>> divs(vmax + 1);
    for (int d = 1; d <= vmax; d++) {
        for (int m = d; m <= vmax; m += d) {
            if (present[m]) divs[m].push_back(d);
        }
    }

    vector<int> mult_sum(vmax + 1, 0);
    vector<char> active(n + 1, 0);
    long long ordered_sum = 0; // sum_{d} f[d] * mult_sum[d]^2
    long long diag_sum = 0;    // sum_{i active} a[i]
    const long long INV2 = (MOD + 1) / 2;

    while (q--) {
        int x;
        cin >> x;
        int v = a[x];
        if (!active[x]) {
            active[x] = 1;
            diag_sum = (diag_sum + v) % MOD;
            for (int d : divs[v]) {
                long long old_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                mult_sum[d] += v;
                if (mult_sum[d] >= MOD) mult_sum[d] -= MOD;
                long long new_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                long long diff = new_sq - old_sq;
                if (diff < 0) diff += MOD;
                ordered_sum = (ordered_sum + diff * f[d]) % MOD;
            }
        } else {
            active[x] = 0;
            diag_sum = (diag_sum - v + MOD) % MOD;
            for (int d : divs[v]) {
                long long old_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                mult_sum[d] -= v;
                if (mult_sum[d] < 0) mult_sum[d] += MOD;
                long long new_sq = 1LL * mult_sum[d] * mult_sum[d] % MOD;
                long long diff = new_sq - old_sq;
                if (diff < 0) diff += MOD;
                ordered_sum = (ordered_sum + diff * f[d]) % MOD;
            }
        }
        long long ans = (ordered_sum - diag_sum + MOD) % MOD * INV2 % MOD;
        cout << ans << '\n';
    }
    return 0;
}
"""

BRUTE_108 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<int> pos_in_act(n + 1, -1), act;
    long long ans = 0;
    while (q--) {
        int x;
        cin >> x;
        long long vx = a[x];
        if (pos_in_act[x] == -1) {
            for (int y : act) {
                long long vy = a[y];
                long long l = (vx / std::gcd(vx, vy)) % MOD * (vy % MOD) % MOD;
                ans += l;
                if (ans >= MOD) ans -= MOD;
            }
            pos_in_act[x] = (int)act.size();
            act.push_back(x);
        } else {
            int idx = pos_in_act[x];
            int last = act.back();
            act[idx] = last;
            pos_in_act[last] = idx;
            act.pop_back();
            pos_in_act[x] = -1;
            for (int y : act) {
                long long vy = a[y];
                long long l = (vx / std::gcd(vx, vy)) % MOD * (vy % MOD) % MOD;
                ans -= l;
                if (ans < 0) ans += MOD;
            }
        }
        cout << ans << '\n';
    }
    return 0;
}
"""


def gen_108(n: int, q: int, vmax: int, seed: int, bulk_activate: bool = False) -> str:
    r = random.Random(seed)
    a = [r.randint(1, vmax) for _ in range(n)]
    lines = [f"{n} {q}", " ".join(map(str, a))]
    if bulk_activate:
        # Activate half the elements first so active set stays >= N/2
        half = min(n, q // 2)
        perm = list(range(1, n + 1))
        r.shuffle(perm)
        for i in range(half):
            lines.append(str(perm[i]))
        for _ in range(q - half):
            lines.append(str(r.randint(1, n)))
    else:
        for _ in range(q):
            lines.append(str(r.randint(1, n)))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_109 — Online Critical Causeways (Online 2ECC Bridge-Tree Contraction)
# ---------------------------------------------------------------------------
FAST_109 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> dsu_cc(n + 1), sz_cc(n + 1, 1), bridges(n + 1, 0);
    vector<int> dsu_2ecc(n + 1), par(n + 1, 0), vis(n + 1, 0);
    iota(dsu_cc.begin(), dsu_cc.end(), 0);
    iota(dsu_2ecc.begin(), dsu_2ecc.end(), 0);
    int timer = 0;

    auto find_2ecc = [&](auto& self, int u) -> int {
        if (u == 0) return 0;
        return dsu_2ecc[u] == u ? u : dsu_2ecc[u] = self(self, dsu_2ecc[u]);
    };

    auto find_cc = [&](auto& self, int u) -> int {
        u = find_2ecc(find_2ecc, u);
        return dsu_cc[u] == u ? u : dsu_cc[u] = self(self, dsu_cc[u]);
    };

    auto make_root = [&](int u) {
        u = find_2ecc(find_2ecc, u);
        int root = u, child = 0;
        while (u != 0) {
            int p = find_2ecc(find_2ecc, par[u]);
            par[u] = child;
            dsu_cc[u] = root;
            child = u;
            u = p;
        }
        sz_cc[root] = sz_cc[child];
        bridges[root] = bridges[child];
    };

    vector<int> path_a, path_b;
    auto merge_path = [&](int a, int b) {
        ++timer;
        path_a.clear();
        path_b.clear();
        int lca = 0;
        while (lca == 0) {
            if (a != 0) {
                a = find_2ecc(find_2ecc, a);
                path_a.push_back(a);
                if (vis[a] == timer) {
                    lca = a;
                    break;
                }
                vis[a] = timer;
                a = par[a];
            }
            if (b != 0) {
                b = find_2ecc(find_2ecc, b);
                path_b.push_back(b);
                if (vis[b] == timer) {
                    lca = b;
                    break;
                }
                vis[b] = timer;
                b = par[b];
            }
        }
        int rc = find_cc(find_cc, lca);
        for (int v : path_a) {
            dsu_2ecc[v] = lca;
            if (v == lca) break;
            bridges[rc]--;
        }
        for (int v : path_b) {
            dsu_2ecc[v] = lca;
            if (v == lca) break;
            bridges[rc]--;
        }
    };

    auto add_edge = [&](int a, int b) {
        a = find_2ecc(find_2ecc, a);
        b = find_2ecc(find_2ecc, b);
        if (a == b) return;
        int ca = find_cc(find_cc, a);
        int cb = find_cc(find_cc, b);
        if (ca != cb) {
            if (sz_cc[ca] > sz_cc[cb]) {
                swap(a, b);
                swap(ca, cb);
            }
            make_root(a);
            par[a] = b;
            dsu_cc[a] = b;
            sz_cc[cb] += sz_cc[a];
            bridges[cb] += bridges[a] + 1;
        } else {
            merge_path(a, b);
        }
    };

    long long last = 0;
    while (q--) {
        int tp, rx, ry;
        cin >> tp >> rx >> ry;
        int u = (int)((rx + last) % n) + 1;
        int v = (int)((ry + last) % n) + 1;
        if (tp == 1) {
            add_edge(u, v);
        } else {
            int cu = find_cc(find_cc, u);
            int cv = find_cc(find_cc, v);
            if (cu != cv) {
                cout << "-1 0\n";
            } else {
                int same2 = (find_2ecc(find_2ecc, u) == find_2ecc(find_2ecc, v)) ? 1 : 0;
                int br = bridges[cu];
                cout << same2 << ' ' << br << '\n';
                last = (last + same2 + br) % n;
            }
        }
    }
    return 0;
}
"""

BRUTE_109 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    int eid_cnt = 0;

    vector<int> tin(n + 1, 0), low(n + 1, 0), comp2(n + 1, 0);
    vector<char> is_bridge;
    long long last = 0;

    while (q--) {
        int tp, rx, ry;
        cin >> tp >> rx >> ry;
        int u = (int)((rx + last) % n) + 1;
        int v = (int)((ry + last) % n) + 1;
        if (tp == 1) {
            int id = eid_cnt++;
            adj[u].push_back({v, id});
            adj[v].push_back({u, id});
        } else {
            is_bridge.assign(eid_cnt, 0);
            // Iterative DFS from u to find component and bridges
            vector<int> visited;
            struct Frame {
                int u, peid, idx;
            };
            vector<Frame> stk;
            int timer = 0, br_cnt = 0;
            tin[u] = low[u] = ++timer;
            visited.push_back(u);
            stk.push_back({u, -1, 0});
            while (!stk.empty()) {
                auto& fr = stk.back();
                int x = fr.u;
                if (fr.idx < (int)adj[x].size()) {
                    auto [y, id] = adj[x][fr.idx++];
                    if (id == fr.peid) continue;
                    if (!tin[y]) {
                        tin[y] = low[y] = ++timer;
                        visited.push_back(y);
                        stk.push_back({y, id, 0});
                    } else {
                        low[x] = min(low[x], tin[y]);
                    }
                } else {
                    int pe = fr.peid;
                    stk.pop_back();
                    if (!stk.empty()) {
                        int p = stk.back().u;
                        low[p] = min(low[p], low[x]);
                        if (low[x] > tin[p]) {
                            is_bridge[pe] = 1;
                            br_cnt++;
                        }
                    }
                }
            }
            if (!tin[v]) {
                cout << "-1 0\n";
            } else {
                // Check if v reachable from u without crossing bridge
                queue<int> bfs;
                bfs.push(u);
                comp2[u] = 1;
                while (!bfs.empty()) {
                    int x = bfs.front();
                    bfs.pop();
                    for (auto [y, id] : adj[x]) {
                        if (!is_bridge[id] && !comp2[y]) {
                            comp2[y] = 1;
                            bfs.push(y);
                        }
                    }
                }
                int same2 = comp2[v] ? 1 : 0;
                cout << same2 << ' ' << br_cnt << '\n';
                last = (last + same2 + br_cnt) % n;
            }
            for (int x : visited) {
                tin[x] = low[x] = comp2[x] = 0;
            }
        }
    }
    return 0;
}
"""


def gen_109(n: int, q: int, seed: int) -> str:
    r = random.Random(seed)
    lines = [f"{n} {q}"]
    # First ~45% updates build a giant component with bridges & cycles regardless of shift
    warmup = int(q * 0.45)
    for i in range(q):
        if i < warmup or r.random() < 0.55:
            tp = 1
        else:
            tp = 2
        rx = r.randrange(n)
        ry = r.randrange(n)
        lines.append(f"{tp} {rx} {ry}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_110 — Exact-K Allowed-Cycle Permutations (Poly Inv + Ln + Exp via NTT)
# ---------------------------------------------------------------------------
FAST_110 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;
static const int G = 3;

long long mod_pow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

void ntt(vector<int>& a, bool inv) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = mod_pow(G, (MOD - 1) / len);
        if (inv) wlen = mod_pow(wlen, MOD - 2);
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            int half = len >> 1;
            for (int j = 0; j < half; j++) {
                int u = a[i + j];
                int v = (int)(a[i + j + half] * w % MOD);
                a[i + j] = u + v < MOD ? u + v : u + v - MOD;
                a[i + j + half] = u - v >= 0 ? u - v : u - v + MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (inv) {
        long long inv_n = mod_pow(n, MOD - 2);
        for (int& x : a) x = (int)(x * inv_n % MOD);
    }
}

vector<int> poly_mul(const vector<int>& a, const vector<int>& b) {
    if (a.empty() || b.empty()) return {};
    int need = (int)a.size() + (int)b.size() - 1;
    int sz = 1;
    while (sz < need) sz <<= 1;
    vector<int> fa(sz, 0), fb(sz, 0);
    copy(a.begin(), a.end(), fa.begin());
    copy(b.begin(), b.end(), fb.begin());
    ntt(fa, false);
    ntt(fb, false);
    for (int i = 0; i < sz; i++) fa[i] = (int)((long long)fa[i] * fb[i] % MOD);
    ntt(fa, true);
    fa.resize(need);
    return fa;
}

vector<int> poly_inv(const vector<int>& a, int n) {
    vector<int> b = {(int)mod_pow(a[0], MOD - 2)};
    int k = 1;
    while (k < n) {
        k <<= 1;
        int sz = k << 1;
        vector<int> fa(sz, 0), fb(sz, 0);
        int take = min((int)a.size(), k);
        for (int i = 0; i < take; i++) fa[i] = a[i];
        for (int i = 0; i < (int)b.size(); i++) fb[i] = b[i];
        ntt(fa, false);
        ntt(fb, false);
        for (int i = 0; i < sz; i++) {
            long long term = (2 - 1LL * fa[i] * fb[i] % MOD + MOD) % MOD;
            fb[i] = (int)(1LL * fb[i] * term % MOD);
        }
        ntt(fb, true);
        fb.resize(k);
        b.swap(fb);
    }
    b.resize(n);
    return b;
}

vector<int> poly_deriv(const vector<int>& a) {
    if ((int)a.size() <= 1) return {0};
    vector<int> r(a.size() - 1);
    for (size_t i = 1; i < a.size(); i++) r[i - 1] = (int)(1LL * a[i] * i % MOD);
    return r;
}

vector<int> poly_integ(const vector<int>& a, const vector<int>& invs) {
    vector<int> r(a.size() + 1, 0);
    for (size_t i = 0; i < a.size(); i++) r[i + 1] = (int)(1LL * a[i] * invs[i + 1] % MOD);
    return r;
}

vector<int> poly_ln(const vector<int>& a, int n, const vector<int>& invs) {
    // Assumes a[0] == 1
    vector<int> d = poly_deriv(a);
    if ((int)d.size() > n) d.resize(n);
    vector<int> ia = poly_inv(a, n);
    vector<int> prod = poly_mul(d, ia);
    if ((int)prod.size() > n - 1) prod.resize(max(0, n - 1));
    vector<int> res = poly_integ(prod, invs);
    res.resize(n, 0);
    return res;
}

vector<int> poly_exp(const vector<int>& a, int n, const vector<int>& invs) {
    // Assumes a[0] == 0
    vector<int> b = {1};
    int k = 1;
    while (k < n) {
        k <<= 1;
        vector<int> lb = poly_ln(b, k, invs);
        vector<int> rhs(k, 0);
        int take = min((int)a.size(), k);
        for (int i = 0; i < k; i++) {
            int av = (i < take ? a[i] : 0);
            rhs[i] = (av - lb[i] + MOD) % MOD;
        }
        rhs[0] = (rhs[0] + 1) % MOD;
        vector<int> nb = poly_mul(b, rhs);
        nb.resize(k, 0);
        b.swap(nb);
    }
    b.resize(n);
    return b;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, m;
    if (!(cin >> n >> k >> m)) return 0;
    vector<int> s(m);
    for (int i = 0; i < m; i++) cin >> s[i];
    sort(s.begin(), s.end());
    s.erase(unique(s.begin(), s.end()), s.end());
    m = (int)s.size();

    int s0 = s[0];
    if (1LL * k * s0 > n) {
        cout << 0 << '\n';
        return 0;
    }

    vector<int> invs(n + 2, 1);
    for (int i = 2; i <= n + 1; i++) {
        invs[i] = (int)(1LL * (MOD - MOD / i) * invs[MOD % i] % MOD);
    }
    long long fact_n = 1, fact_k = 1;
    for (int i = 1; i <= n; i++) fact_n = fact_n * i % MOD;
    for (int i = 1; i <= k; i++) fact_k = fact_k * i % MOD;

    int deg = n - k * s0;
    int len = deg + 1;
    // B(x) = sum_{s in S} (1/s) x^{s - s0}, normalized by B(0)^{-1} = s0
    vector<int> B(len, 0);
    for (int len_c : s) {
        if (len_c - s0 <= deg) {
            B[len_c - s0] = (int)(1LL * invs[len_c] * s0 % MOD);
        }
    }
    // B[0] == 1; compute exp(k * ln(B)) mod x^len
    vector<int> lnB = poly_ln(B, len, invs);
    for (int i = 0; i < len; i++) lnB[i] = (int)(1LL * lnB[i] * k % MOD);
    vector<int> pB = poly_exp(lnB, len, invs);

    long long coeff = pB[deg] * mod_pow(invs[s0], k) % MOD;
    long long ans = coeff * fact_n % MOD * mod_pow(fact_k, MOD - 2) % MOD;
    cout << ans << '\n';
    return 0;
}
"""

BRUTE_110 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

long long mod_pow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, m;
    cin >> n >> k >> m;
    vector<int> s(m);
    for (int i = 0; i < m; i++) cin >> s[i];
    sort(s.begin(), s.end());
    s.erase(unique(s.begin(), s.end()), s.end());

    vector<int> invs(n + 1, 1);
    for (int i = 2; i <= n; i++) invs[i] = (int)(1LL * (MOD - MOD / i) * invs[MOD % i] % MOD);

    vector<int> base(n + 1, 0), res(n + 1, 0), tmp(n + 1, 0);
    for (int x : s) if (x <= n) base[x] = invs[x];
    res[0] = 1;

    auto mul_trunc = [&](const vector<int>& A, const vector<int>& B) {
        fill(tmp.begin(), tmp.end(), 0);
        for (int i = 0; i <= n; i++) {
            if (!A[i]) continue;
            long long ai = A[i];
            for (int j = 0; i + j <= n; j++) {
                if (B[j]) tmp[i + j] = (int)((tmp[i + j] + ai * B[j]) % MOD);
            }
        }
        return tmp;
    };

    int e = k;
    while (e > 0) {
        if (e & 1) res = mul_trunc(res, base);
        e >>= 1;
        if (e > 0) base = mul_trunc(base, base);
    }

    long long fn = 1, fk = 1;
    for (int i = 1; i <= n; i++) fn = fn * i % MOD;
    for (int i = 1; i <= k; i++) fk = fk * i % MOD;
    cout << 1LL * res[n] * fn % MOD * mod_pow(fk, MOD - 2) % MOD << '\n';
    return 0;
}
"""


def gen_110(n: int, k: int, m: int, min_s: int, seed: int) -> str:
    r = random.Random(seed)
    pool = list(range(min_s, n + 1))
    m = min(m, len(pool))
    s = sorted(r.sample(pool, m))
    # Guarantee s[0] = min_s so k * min_s <= n can be satisfied
    s[0] = min_s
    return f"{n} {k} {m}\n" + " ".join(map(str, s)) + "\n"


def make_specs() -> list[ProblemSpec]:
    return [
        ProblemSpec(
            pid="synth_106",
            title="Patrol Checkpoints",
            rating=2800,
            tags=["flows", "trees", "hld", "data structures"],
            notes="CF 786E style bipartite min-cut on tree paths accelerated via HLD + Segment Tree Dinic.",
            story_and_spec="""A kingdom has N cities connected by N - 1 bidirectional roads forming a tree. Fortifying road i with a checkpoint costs C_i gold.

Intelligence reports M smuggling routes. Smuggler j travels along the unique simple path between cities U_j and V_j. If at least one road on that path is fortified, smuggler j is intercepted; otherwise the kingdom suffers a loss of P_j gold.

Find the minimum possible total sum of checkpoint fortification costs plus losses from unintercepted smugglers.

Input
The first line contains N and M.
Each of the next N - 1 lines contains U_i, V_i, and C_i describing a road.
Each of the next M lines contains U_j, V_j, and P_j describing a smuggler route.

Output
Print the minimum total cost.

Constraints
2 <= N <= 20000
1 <= M <= 20000
1 <= C_i, P_j <= 10^6""",
            samples=[
                "5 3\n1 2 5\n2 3 4\n2 4 6\n4 5 3\n1 3 10\n3 5 8\n1 4 2\n",
                "4 2\n1 2 10\n2 3 10\n3 4 10\n1 4 7\n2 2 100\n",
            ],
            small_inputs=[gen_106(320, 320, s, chain=(s == 1)) for s in range(1, 5)],
            large_inputs=[
                gen_106(20000, 20000, 101, chain=True),
                gen_106(20000, 20000, 102, chain=False),
                gen_106(18000, 20000, 103, chain=True),
                gen_106(20000, 18000, 104, chain=False),
            ],
            fast_cpp=FAST_106,
            brute_cpp=BRUTE_106,
        ),
        ProblemSpec(
            pid="synth_107",
            title="Minkowski Caravan Envelope",
            rating=2900,
            tags=["geometry", "convex hull", "binary search"],
            notes="Minkowski sum of N convex hulls via global CCW edge sort + upper/lower chain binary search.",
            story_and_spec="""A trade guild coordinates N regional chapters. Chapter i controls m_i outpost locations (X_{i,j}, Y_{i,j}) in the 2D plane. The guild must select exactly one outpost p_i from each chapter i, forming the combined vector S = p_1 + p_2 + ... + p_N.

You are given Q trade-wind scenarios. Scenario q specifies an integer direction vector (A_q, B_q) (not both zero) and asks for the maximum possible dot product A_q * S_x + B_q * S_y achievable over all valid choices of (p_1, ..., p_N).

Input
The first line contains N and Q.
Then N blocks follow. Block i begins with a line containing m_i, followed by m_i lines each containing X_{i,j} and Y_{i,j}.
After all N blocks, Q lines follow, each containing A_q and B_q.

Output
For each scenario print the maximum dot product on its own line.

Constraints
1 <= N <= 40000
1 <= sum of m_i <= 150000
1 <= Q <= 150000
-10^6 <= X_{i,j}, Y_{i,j}, A_q, B_q <= 10^6
(A_q, B_q) != (0, 0)""",
            samples=[
                "2 4\n3\n0 0\n2 0\n0 3\n2\n1 1\n4 -1\n1 1\n1 0\n0 1\n-1 -1\n",
                "3 2\n1\n5 -2\n2\n0 0\n0 4\n3\n-1 0\n1 0\n0 2\n2 3\n-3 1\n",
            ],
            small_inputs=[gen_107(120, 2200, 2200, s) for s in range(1, 5)],
            large_inputs=[
                gen_107(25000, 150000, 150000, 201),
                gen_107(40000, 150000, 150000, 202),
                gen_107(5000, 150000, 150000, 203),
                gen_107(15000, 145000, 150000, 204),
            ],
            fast_cpp=FAST_107,
            brute_cpp=BRUTE_107,
        ),
        ProblemSpec(
            pid="synth_108",
            title="Harmonic Canister Shelf",
            rating=2800,
            tags=["number theory", "mobius inversion", "math"],
            notes="Dynamic multiset sum of pairwise lcm(A_i, A_j) via Dirichlet convolution f = mu * id^{-1}.",
            story_and_spec="""A laboratory shelf has N numbered slots 1 .. N. Slot i holds a canister stamped with frequency A_i. Initially all N slots are inactive.

Process Q toggle events. Each event gives a slot index x (1 <= x <= N): if slot x is currently inactive it becomes active, and if it is active it becomes inactive.

After each event, compute the sum of lcm(A_i, A_j) over all unordered pairs of distinct currently active slots {i, j} (1 <= i < j <= N), modulo 998244353 (where lcm(a, b) = a * b / gcd(a, b)). If fewer than two slots are active, the sum is 0.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains a single integer x.

Output
After each event print the pairwise LCM sum modulo 998244353 on its own line.

Constraints
1 <= N <= 150000
1 <= Q <= 180000
1 <= A_i <= 300000""",
            samples=[
                "4 6\n6 10 15 6\n1\n2\n3\n4\n2\n1\n",
                "3 4\n12 18 12\n1\n2\n3\n1\n",
            ],
            small_inputs=[gen_108(2000, 2200, 50000, s, bulk_activate=True) for s in range(1, 5)],
            large_inputs=[
                gen_108(150000, 180000, 300000, 301, bulk_activate=True),
                gen_108(150000, 180000, 300000, 302, bulk_activate=False),
                gen_108(140000, 180000, 288288, 303, bulk_activate=True),
                gen_108(150000, 175000, 300000, 304, bulk_activate=True),
            ],
            fast_cpp=FAST_108,
            brute_cpp=BRUTE_108,
        ),
        ProblemSpec(
            pid="synth_109",
            title="Online Critical Causeways",
            rating=2800,
            tags=["graphs", "dsu", "dfs and similar", "trees"],
            notes="Strictly online 2-edge-connected components and per-component bridge counts via two-DSU bridge-tree contraction.",
            story_and_spec="""An archipelago has N islands numbered 1 to N and initially 0 causeways.

Process Q online events. Maintain an integer `last`, initially 0. Each event has a type t in {1, 2} and two encoded integers rx, ry (0 <= rx, ry < N). Decode the islands as:
u = (rx + last) mod N + 1,
v = (ry + last) mod N + 1.

- Type 1 : build a new bidirectional causeway between u and v (parallel causeways and self-loops are allowed).
- Type 2 :
  - If u and v are in different connected components, print `-1 0` (and leave `last` unchanged).
  - Otherwise, let `same` be 1 if u and v belong to the same 2-edge-connected component (i.e., removing any single causeway in the archipelago does NOT disconnect u from v) and 0 otherwise, and let `br` be the total number of bridge causeways currently inside u's connected component. Print `same br`, then update `last <- (last + same + br) mod N`.

Input
The first line contains N and Q.
Each of the next Q lines contains t, rx, and ry.

Output
For each type-2 event print two space-separated integers on their own line.

Constraints
1 <= N <= 150000
1 <= Q <= 200000
t in {1, 2}, 0 <= rx, ry < N""",
            samples=[
                "5 8\n1 0 1\n1 1 2\n2 0 2\n1 2 0\n2 0 2\n1 0 3\n2 0 1\n2 3 4\n",
                "4 5\n2 0 0\n1 0 1\n1 0 1\n2 0 1\n2 0 2\n",
            ],
            small_inputs=[gen_109(1800, 2200, s) for s in range(1, 5)],
            large_inputs=[
                gen_109(100000, 200000, 401),
                gen_109(150000, 200000, 402),
                gen_109(120000, 200000, 403),
                gen_109(150000, 190000, 404),
            ],
            fast_cpp=FAST_109,
            brute_cpp=BRUTE_109,
        ),
        ProblemSpec(
            pid="synth_110",
            title="Constrained Cycle Permutations",
            rating=2900,
            tags=["fft", "ntt", "polynomials", "combinatorics"],
            notes="EGF [x^N] A(x)^K / K! * N! via Newton-iteration polynomial inv, ln, and exp with NTT.",
            story_and_spec="""You are given positive integers N and K, and a set S of M allowed cycle lengths.

Count how many permutations of {1, 2, ..., N} decompose into EXACTLY K disjoint directed cycles such that the length of every cycle belongs to S. Print the count modulo 998244353.

Input
The first line contains N, K, and M.
The second line contains M integers s_1 ... s_M describing S.

Output
Print the number of valid permutations modulo 998244353.

Constraints
1 <= K <= N <= 100000
1 <= M <= N
1 <= s_i <= N""",
            samples=[
                "4 2 3\n1 2 3\n",
                "6 2 2\n2 4\n",
            ],
            small_inputs=[
                gen_110(1400, 400, 500, 1, 1),
                gen_110(1500, 300, 800, 2, 2),
                gen_110(1200, 700, 400, 2, 3),
                gen_110(1500, 1, 1000, 1, 4),
            ],
            large_inputs=[
                gen_110(100000, 30000, 50000, 1, 501),
                gen_110(100000, 20000, 40000, 2, 502),
                gen_110(95000, 40000, 30000, 2, 503),
                gen_110(100000, 15000, 80000, 3, 504),
            ],
            fast_cpp=FAST_110,
            brute_cpp=BRUTE_110,
        ),
    ]


if __name__ == "__main__":
    build_and_verify_batch(make_specs())
