"""Batch 23: synth_111 .. synth_115 (Grandmaster Wave VII: CF 2700-2900)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from builder import ProblemSpec, build_and_verify_batch


# ---------------------------------------------------------------------------
# synth_111 — Dynamic Tree Diameter (CEOI 2019 Euler-Tour 5-Tuple SegTree)
# ---------------------------------------------------------------------------
FAST_111 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 60);
static const long long POS_INF = (1LL << 60);

struct Node {
    long long mx, mn, lm, mr, lmr, lz;
};

static inline Node merge_node(const Node& a, const Node& b) {
    Node c;
    c.mx = max(a.mx, b.mx);
    c.mn = min(a.mn, b.mn);
    c.lm = max({a.lm, b.lm, a.mx - 2 * b.mn});
    c.mr = max({a.mr, b.mr, -2 * a.mn + b.mx});
    c.lmr = max({a.lmr, b.lmr, a.lm + b.mx, a.mx + b.mr});
    c.lz = 0;
    return c;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    struct Edge {
        int u, v;
        long long w;
    };
    vector<Edge> edges(n);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= n - 1; i++) {
        cin >> edges[i].u >> edges[i].v >> edges[i].w;
        adj[edges[i].u].push_back({edges[i].v, i});
        adj[edges[i].v].push_back({edges[i].u, i});
    }

    // Iterative full Euler tour of length M = 2*n - 1
    int M = 2 * n - 1;
    vector<long long> init_d(M + 1, 0), dep_w(n + 1, 0);
    vector<int> first_pos(n + 1, 0), last_pos(n + 1, 0), child_end(n, 0);
    struct Frame {
        int u, p, idx;
    };
    vector<Frame> stk;
    stk.reserve(n);
    stk.push_back({1, 0, 0});
    int timer = 0;
    first_pos[1] = last_pos[1] = ++timer;
    init_d[timer] = 0;

    while (!stk.empty()) {
        auto& fr = stk.back();
        int u = fr.u;
        if (fr.idx < (int)adj[u].size()) {
            auto [v, eid] = adj[u][fr.idx++];
            if (v == fr.p) continue;
            child_end[eid] = v;
            dep_w[v] = dep_w[u] + edges[eid].w;
            first_pos[v] = last_pos[v] = ++timer;
            init_d[timer] = dep_w[v];
            stk.push_back({v, u, 0});
        } else {
            int p = fr.p;
            stk.pop_back();
            if (p != 0) {
                last_pos[p] = ++timer;
                init_d[timer] = dep_w[p];
            }
        }
    }

    vector<Node> tr(4 * M + 5);

    auto apply_add = [&](int u, long long d) {
        tr[u].mx += d;
        tr[u].mn += d;
        tr[u].lm -= d;
        tr[u].mr -= d;
        // tr[u].lmr is invariant under uniform shift +d
        tr[u].lz += d;
    };

    auto push = [&](int u) {
        if (tr[u].lz != 0) {
            apply_add(u << 1, tr[u].lz);
            apply_add(u << 1 | 1, tr[u].lz);
            tr[u].lz = 0;
        }
    };

    auto build = [&](auto& self, int u, int l, int r) -> void {
        if (l == r) {
            long long d = init_d[l];
            tr[u] = {d, d, -d, -d, 0, 0};
            return;
        }
        int m = (l + r) >> 1;
        self(self, u << 1, l, m);
        self(self, u << 1 | 1, m + 1, r);
        tr[u] = merge_node(tr[u << 1], tr[u << 1 | 1]);
    };

    auto update = [&](auto& self, int u, int l, int r, int ql, int qr, long long d) -> void {
        if (ql <= l && r <= qr) {
            apply_add(u, d);
            return;
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, d);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, d);
        long long saved_lz = 0;
        tr[u] = merge_node(tr[u << 1], tr[u << 1 | 1]);
        tr[u].lz = saved_lz;
    };

    build(build, 1, 1, M);

    while (q--) {
        int eid;
        long long nw;
        cin >> eid >> nw;
        long long delta = nw - edges[eid].w;
        edges[eid].w = nw;
        int c = child_end[eid];
        update(update, 1, 1, M, first_pos[c], last_pos[c], delta);
        cout << tr[1].lmr << '\n';
    }
    return 0;
}
"""

BRUTE_111 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> eu(n), ev(n);
    vector<long long> ew(n);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= n - 1; i++) {
        cin >> eu[i] >> ev[i] >> ew[i];
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }

    vector<int> par(n + 1), que(n + 1);
    vector<long long> dist(n + 1);
    auto farthest = [&](int src) -> pair<int, long long> {
        int head = 0, tail = 0;
        que[tail++] = src;
        par[src] = 0;
        dist[src] = 0;
        int best_u = src;
        long long best_d = 0;
        while (head < tail) {
            int u = que[head++];
            if (dist[u] > best_d) {
                best_d = dist[u];
                best_u = u;
            }
            for (auto [v, id] : adj[u]) {
                if (v != par[u]) {
                    par[v] = u;
                    dist[v] = dist[u] + ew[id];
                    que[tail++] = v;
                }
            }
        }
        return {best_u, best_d};
    };

    while (q--) {
        int eid;
        long long nw;
        cin >> eid >> nw;
        ew[eid] = nw;
        auto [x, _] = farthest(1);
        auto [y, diam] = farthest(x);
        cout << diam << '\n';
    }
    return 0;
}
"""


def gen_111(n: int, q: int, seed: int, chain: bool = False) -> str:
    r = random.Random(seed)
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    lines = [f"{n} {q}"]
    for i in range(1, n):
        p = perm[i - 1] if chain else perm[r.randint(max(0, i - 8), i - 1)]
        w = r.randint(1, 10**9)
        lines.append(f"{p} {perm[i]} {w}")
    for _ in range(q):
        eid = r.randint(1, n - 1)
        nw = r.randint(1, 10**9)
        lines.append(f"{eid} {nw}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_112 — Bitwise AND/OR SegTree Beats with Range Max & Range Sum
# ---------------------------------------------------------------------------
FAST_112 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int FULL = (1 << 30) - 1;

struct Node {
    int all_and, all_or, mx;
    long long sum;
    int lz_and, lz_or;
    int len;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<Node> tr(4 * n + 5);

    auto apply_uniform = [&](int u, int A, int O) {
        int new_and = (tr[u].all_and & A) | O;
        long long delta = (long long)new_and - (long long)tr[u].all_and;
        tr[u].all_and = new_and;
        tr[u].all_or = (tr[u].all_or & A) | O;
        tr[u].mx = (int)(tr[u].mx + delta);
        tr[u].sum += delta * tr[u].len;
        tr[u].lz_and &= A;
        tr[u].lz_or = (tr[u].lz_or & A) | O;
    };

    auto push = [&](int u) {
        if (tr[u].lz_and != FULL || tr[u].lz_or != 0) {
            apply_uniform(u << 1, tr[u].lz_and, tr[u].lz_or);
            apply_uniform(u << 1 | 1, tr[u].lz_and, tr[u].lz_or);
            tr[u].lz_and = FULL;
            tr[u].lz_or = 0;
        }
    };

    auto pull = [&](int u) {
        const auto& L = tr[u << 1];
        const auto& R = tr[u << 1 | 1];
        tr[u].all_and = L.all_and & R.all_and;
        tr[u].all_or = L.all_or | R.all_or;
        tr[u].mx = max(L.mx, R.mx);
        tr[u].sum = L.sum + R.sum;
    };

    auto build = [&](auto& self, int u, int l, int r) -> void {
        tr[u].lz_and = FULL;
        tr[u].lz_or = 0;
        tr[u].len = r - l + 1;
        if (l == r) {
            tr[u].all_and = tr[u].all_or = tr[u].mx = a[l];
            tr[u].sum = a[l];
            return;
        }
        int m = (l + r) >> 1;
        self(self, u << 1, l, m);
        self(self, u << 1 | 1, m + 1, r);
        pull(u);
    };

    auto upd_and = [&](auto& self, int u, int l, int r, int ql, int qr, int x) -> void {
        int clr = FULL ^ x;
        if ((tr[u].all_or & clr) == 0) return;
        if (ql <= l && r <= qr) {
            int diff = tr[u].all_or ^ tr[u].all_and;
            if ((diff & clr) == 0) {
                apply_uniform(u, x, 0);
                return;
            }
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, x);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, x);
        pull(u);
    };

    auto upd_or = [&](auto& self, int u, int l, int r, int ql, int qr, int x) -> void {
        if (((FULL ^ tr[u].all_and) & x) == 0) return;
        if (ql <= l && r <= qr) {
            int diff = tr[u].all_or ^ tr[u].all_and;
            if ((diff & x) == 0) {
                apply_uniform(u, FULL, x);
                return;
            }
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, x);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, x);
        pull(u);
    };

    auto query = [&](auto& self, int u, int l, int r, int ql, int qr) -> pair<int, long long> {
        if (ql <= l && r <= qr) return {tr[u].mx, tr[u].sum};
        push(u);
        int m = (l + r) >> 1;
        pair<int, long long> res{-1, 0};
        if (ql <= m) {
            auto lv = self(self, u << 1, l, m, ql, qr);
            res.first = max(res.first, lv.first);
            res.second += lv.second;
        }
        if (qr > m) {
            auto rv = self(self, u << 1 | 1, m + 1, r, ql, qr);
            res.first = max(res.first, rv.first);
            res.second += rv.second;
        }
        return res;
    };

    build(build, 1, 1, n);

    while (q--) {
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            int x;
            cin >> x;
            upd_and(upd_and, 1, 1, n, l, r, x);
        } else if (tp == 2) {
            int x;
            cin >> x;
            upd_or(upd_or, 1, 1, n, l, r, x);
        } else {
            auto [mx, sm] = query(query, 1, 1, n, l, r);
            cout << mx << ' ' << sm << '\n';
        }
    }
    return 0;
}
"""

BRUTE_112 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            int x;
            cin >> x;
            for (int i = l; i <= r; i++) a[i] &= x;
        } else if (tp == 2) {
            int x;
            cin >> x;
            for (int i = l; i <= r; i++) a[i] |= x;
        } else {
            int mx = 0;
            long long sm = 0;
            for (int i = l; i <= r; i++) {
                mx = max(mx, a[i]);
                sm += a[i];
            }
            cout << mx << ' ' << sm << '\n';
        }
    }
    return 0;
}
"""


def gen_112(n: int, q: int, seed: int, wide: bool = False) -> str:
    r = random.Random(seed)
    full = (1 << 30) - 1
    a = [r.randint(0, full) for _ in range(n)]
    lines = [f"{n} {q}", " ".join(map(str, a))]
    for _ in range(q):
        tp = r.choices([1, 2, 3], weights=[35, 35, 30])[0]
        if wide and r.random() < 0.8:
            ql = r.randint(1, max(1, n // 10))
            qr = r.randint(max(ql, n - n // 10), n)
        else:
            u = r.randint(1, n)
            v = r.randint(1, n)
            ql, qr = min(u, v), max(u, v)
        if tp == 1:
            # Clear 1-4 random bits so entropy stays rich across 150k ops
            mask = full
            for _ in range(r.randint(1, 4)):
                mask &= ~(1 << r.randrange(30))
            lines.append(f"1 {ql} {qr} {mask}")
        elif tp == 2:
            mask = 0
            for _ in range(r.randint(1, 4)):
                mask |= 1 << r.randrange(30)
            lines.append(f"2 {ql} {qr} {mask}")
        else:
            lines.append(f"3 {ql} {qr}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_113 — K-th Smallest & Sum of K Smallest Tree Paths (Centroid + BS)
# ---------------------------------------------------------------------------
FAST_113 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long K;
    if (!(cin >> n >> K)) return 0;
    vector<vector<pair<int, long long>>> adj(n + 1);
    long long max_edge = 0;
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        max_edge = max(max_edge, w);
    }

    vector<vector<long long>> cent_dists, sub_dists;
    cent_dists.reserve(n);
    sub_dists.reserve(n);

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), par_t(n + 1, 0), nodes;
    vector<long long> d_tmp(n + 1, 0);
    nodes.reserve(n);

    auto build_cd = [&](auto& self, int entry) -> void {
        nodes.clear();
        par_t[entry] = 0;
        nodes.push_back(entry);
        for (size_t i = 0; i < nodes.size(); i++) {
            int u = nodes[i];
            sz[u] = 1;
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_t[u]) {
                    par_t[v] = u;
                    nodes.push_back(v);
                }
            }
        }
        int tot = (int)nodes.size();
        if (tot <= 1) {
            dead[entry] = 1;
            return;
        }
        for (int i = tot - 1; i >= 0; i--) {
            int u = nodes[i];
            if (par_t[u]) sz[par_t[u]] += sz[u];
        }
        int cent = entry;
        for (int u : nodes) {
            int mx = tot - sz[u];
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_t[u]) mx = max(mx, sz[v]);
            }
            if (mx * 2 <= tot) {
                cent = u;
                break;
            }
        }

        dead[cent] = 1;
        vector<long long> all_c;
        all_c.reserve(tot);
        all_c.push_back(0);

        for (auto [nb, w0] : adj[cent]) {
            if (dead[nb]) continue;
            nodes.clear();
            par_t[nb] = cent;
            d_tmp[nb] = w0;
            nodes.push_back(nb);
            for (size_t i = 0; i < nodes.size(); i++) {
                int u = nodes[i];
                for (auto [v, w] : adj[u]) {
                    if (!dead[v] && v != par_t[u]) {
                        par_t[v] = u;
                        d_tmp[v] = d_tmp[u] + w;
                        nodes.push_back(v);
                    }
                }
            }
            vector<long long> sv;
            sv.reserve(nodes.size());
            for (int u : nodes) {
                sv.push_back(d_tmp[u]);
                all_c.push_back(d_tmp[u]);
            }
            sort(sv.begin(), sv.end());
            if (sv.size() >= 2) sub_dists.push_back(move(sv));
        }
        sort(all_c.begin(), all_c.end());
        if (all_c.size() >= 2) cent_dists.push_back(move(all_c));

        for (auto [nb, w0] : adj[cent]) {
            if (!dead[nb]) self(self, nb);
        }
    };

    build_cd(build_cd, 1);

    auto count_pairs = [&](long long X) -> long long {
        long long total = 0;
        for (const auto& vec : cent_dists) {
            int r = (int)vec.size() - 1;
            for (int l = 0; l < r; l++) {
                while (r > l && vec[l] + vec[r] > X) r--;
                if (r > l) total += (r - l);
            }
        }
        for (const auto& vec : sub_dists) {
            int r = (int)vec.size() - 1;
            for (int l = 0; l < r; l++) {
                while (r > l && vec[l] + vec[r] > X) r--;
                if (r > l) total -= (r - l);
            }
        }
        return total;
    };

    long long lo = 1, hi = max_edge * (n - 1), kth_d = hi;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_pairs(mid) >= K) {
            kth_d = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    // Sum and count strictly less than kth_d
    auto count_and_sum = [&](const vector<vector<long long>>& lists, long long X) -> pair<long long, long long> {
        long long cnt = 0, sm = 0;
        vector<long long> pref;
        for (const auto& vec : lists) {
            int m = (int)vec.size();
            pref.resize(m + 1);
            pref[0] = 0;
            for (int i = 0; i < m; i++) pref[i + 1] = (pref[i] + vec[i]) % MOD;
            int r = m - 1;
            for (int l = 0; l < r; l++) {
                while (r > l && vec[l] + vec[r] > X) r--;
                if (r > l) {
                    long long c = r - l;
                    cnt += c;
                    long long range_sum = (pref[r + 1] - pref[l + 1] + MOD) % MOD;
                    sm = (sm + c % MOD * (vec[l] % MOD) + range_sum) % MOD;
                }
            }
        }
        return {cnt, sm};
    };

    auto [c1, s1] = count_and_sum(cent_dists, kth_d - 1);
    auto [c2, s2] = count_and_sum(sub_dists, kth_d - 1);
    long long strict_cnt = c1 - c2;
    long long strict_sum = (s1 - s2 + MOD) % MOD;
    long long rem = (K - strict_cnt) % MOD;
    long long ans_sum = (strict_sum + rem * (kth_d % MOD)) % MOD;
    cout << kth_d << ' ' << ans_sum << '\n';
    return 0;
}
"""

BRUTE_113 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

// Memory-safe O(N) RAM solver: binary searches X using N tree BFS traversals per check
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long K;
    cin >> n >> K;
    vector<vector<pair<int, long long>>> adj(n + 1);
    long long max_e = 0;
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        max_e = max(max_e, w);
    }

    vector<int> par(n + 1), que(n + 1);
    vector<long long> dist(n + 1);

    auto count_leq = [&](long long X) -> long long {
        long long cnt = 0;
        for (int s = 1; s <= n; s++) {
            int h = 0, t = 0;
            que[t++] = s;
            par[s] = 0;
            dist[s] = 0;
            while (h < t) {
                int u = que[h++];
                if (u > s && dist[u] <= X) cnt++;
                if (dist[u] >= X) continue;
                for (auto [v, w] : adj[u]) {
                    if (v != par[u] && dist[u] + w <= X) {
                        par[v] = u;
                        dist[v] = dist[u] + w;
                        que[t++] = v;
                    }
                }
            }
        }
        return cnt;
    };

    long long lo = 1, hi = max_e * (n - 1), kth_d = hi;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_leq(mid) >= K) {
            kth_d = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    long long cnt_lt = 0, sum_lt = 0;
    for (int s = 1; s <= n; s++) {
        int h = 0, t = 0;
        que[t++] = s;
        par[s] = 0;
        dist[s] = 0;
        while (h < t) {
            int u = que[h++];
            if (u > s && dist[u] < kth_d) {
                cnt_lt++;
                sum_lt = (sum_lt + dist[u]) % MOD;
            }
            for (auto [v, w] : adj[u]) {
                if (v != par[u]) {
                    par[v] = u;
                    dist[v] = dist[u] + w;
                    que[t++] = v;
                }
            }
        }
    }
    long long ans_sum = (sum_lt + (K - cnt_lt) % MOD * (kth_d % MOD)) % MOD;
    cout << kth_d << ' ' << ans_sum << '\n';
    return 0;
}
"""


def gen_113(n: int, seed: int, high_k: bool = True) -> str:
    r = random.Random(seed)
    tot_pairs = n * (n - 1) // 2
    k = r.randint(max(1, tot_pairs // 2), tot_pairs) if high_k else r.randint(1, tot_pairs)
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    lines = [f"{n} {k}"]
    for i in range(1, n):
        p = perm[r.randint(max(0, i - 6), i - 1)]
        w = r.randint(1, 10**8)
        lines.append(f"{p} {perm[i]} {w}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_114 — Dynamic Line Additions & Deletions (Time-SegTree of Convex Hulls)
# ---------------------------------------------------------------------------
FAST_114 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 62);

struct Line {
    long long a, b;
};

static inline bool bad_upper(const Line& l1, const Line& l2, const Line& l3) {
    // Upper hull of y = a*x + b with a1 < a2 < a3: l2 is redundant iff
    // (b2 - b1)/(a1 - a2) >= (b3 - b2)/(a2 - a3) <=> (b2 - b1)*(a3 - a2) <= (b3 - b2)*(a2 - a1)
    return (__int128)(l2.b - l1.b) * (l3.a - l2.a) <= (__int128)(l3.b - l2.b) * (l2.a - l1.a);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> q)) return 0;

    struct Interval {
        long long a, b;
        int l, r;
    };
    vector<Interval> items;
    vector<int> open_idx;
    vector<pair<int, long long>> queries;

    for (int t = 1; t <= q; t++) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            long long a, b;
            cin >> a >> b;
            open_idx.push_back((int)items.size());
            items.push_back({a, b, t, q});
        } else if (tp == 2) {
            int id;
            cin >> id;
            items[open_idx[id - 1]].r = t - 1;
        } else {
            long long x;
            cin >> x;
            queries.push_back({t, x});
        }
    }

    if (queries.empty()) return 0;

    // Sort lines by slope ascending once so each SegTree node builds hull in O(sz)
    sort(items.begin(), items.end(), [](const Interval& u, const Interval& v) {
        if (u.a != v.a) return u.a < v.a;
        return u.b < v.b;
    });

    vector<vector<Line>> tr(4 * q + 5);
    auto add_line = [&](auto& self, int u, int l, int r, int ql, int qr, const Line& ln) -> void {
        if (ql > qr) return;
        if (ql <= l && r <= qr) {
            auto& h = tr[u];
            if (!h.empty() && h.back().a == ln.a) {
                h.back().b = ln.b; // sorted ascending in b, so ln.b >= h.back().b
            } else {
                while (h.size() >= 2 && bad_upper(h[h.size() - 2], h.back(), ln)) h.pop_back();
                h.push_back(ln);
            }
            return;
        }
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, ln);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, ln);
    };

    for (const auto& it : items) {
        if (it.l <= it.r) add_line(add_line, 1, 1, q, it.l, it.r, {it.a, it.b});
    }

    // Sort queries by x ascending for O(1) amortized pointer advance per node
    int m = (int)queries.size();
    vector<int> qord(m);
    iota(qord.begin(), qord.end(), 0);
    sort(qord.begin(), qord.end(), [&](int i, int j) {
        return queries[i].second < queries[j].second;
    });

    vector<int> ptr(4 * q + 5, 0);
    auto eval_node = [&](int u, long long x) -> long long {
        const auto& h = tr[u];
        if (h.empty()) return NEG_INF;
        int& p = ptr[u];
        while (p + 1 < (int)h.size() && h[p].a * x + h[p].b <= h[p + 1].a * x + h[p + 1].b) {
            p++;
        }
        return h[p].a * x + h[p].b;
    };

    vector<long long> ans(m, NEG_INF);
    for (int idx : qord) {
        int t = queries[idx].first;
        long long x = queries[idx].second;
        int u = 1, l = 1, r = q;
        long long best = NEG_INF;
        while (true) {
            best = max(best, eval_node(u, x));
            if (l == r) break;
            int mid = (l + r) >> 1;
            if (t <= mid) {
                u = u << 1;
                r = mid;
            } else {
                u = u << 1 | 1;
                l = mid + 1;
            }
        }
        ans[idx] = best;
    }

    for (int i = 0; i < m; i++) {
        if (ans[i] == NEG_INF) cout << "EMPTY\n";
        else cout << ans[i] << '\n';
    }
    return 0;
}
"""

BRUTE_114 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 62);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    cin >> q;
    vector<pair<long long, long long>> lines;
    vector<int> pos_in_act;
    vector<int> act;
    for (int t = 1; t <= q; t++) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            long long a, b;
            cin >> a >> b;
            int id = (int)lines.size();
            lines.push_back({a, b});
            pos_in_act.push_back((int)act.size());
            act.push_back(id);
        } else if (tp == 2) {
            int id;
            cin >> id;
            --id;
            int idx = pos_in_act[id];
            int last = act.back();
            act[idx] = last;
            pos_in_act[last] = idx;
            act.pop_back();
            pos_in_act[id] = -1;
        } else {
            long long x;
            cin >> x;
            if (act.empty()) {
                cout << "EMPTY\n";
            } else {
                long long best = NEG_INF;
                for (int id : act) {
                    best = max(best, lines[id].first * x + lines[id].second);
                }
                cout << best << '\n';
            }
        }
    }
    return 0;
}
"""


def gen_114(q: int, seed: int, bulk_lines: bool = False) -> str:
    r = random.Random(seed)
    lines = [str(q)]
    active = []
    next_id = 0
    warmup = int(q * 0.48) if bulk_lines else 0
    for step in range(q):
        if step < warmup:
            tp = 1
        elif bulk_lines:
            tp = r.choices([1, 2, 3], weights=[10, 5, 85])[0]
            if tp == 2 and not active:
                tp = 1
        else:
            tp = r.choices([1, 2, 3], weights=[35, 20, 45])[0]
            if tp == 2 and not active:
                tp = 1
        if tp == 1:
            next_id += 1
            active.append(next_id)
            a = r.randint(-10**6, 10**6)
            b = r.randint(-10**12, 10**12)
            lines.append(f"1 {a} {b}")
        elif tp == 2:
            idx = r.randrange(len(active))
            cid = active[idx]
            active[idx] = active[-1]
            active.pop()
            lines.append(f"2 {cid}")
        else:
            x = r.randint(-10**6, 10**6)
            lines.append(f"3 {x}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_115 — Online Range K-th First-Occurrence Distinct Element (Persistent SegTree)
# ---------------------------------------------------------------------------
FAST_115 = r"""
#include <bits/stdc++.h>
using namespace std;

struct Node {
    int lc, rc, sum;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<Node> tr;
    tr.reserve(n * 40 + 10);
    tr.push_back({0, 0, 0});

    auto upd = [&](auto& self, int prev, int l, int r, int pos, int delta) -> int {
        int u = (int)tr.size();
        tr.push_back(tr[prev]);
        tr[u].sum += delta;
        if (l == r) return u;
        int m = (l + r) >> 1;
        if (pos <= m) {
            int lc = self(self, tr[prev].lc, l, m, pos, delta);
            tr[u].lc = lc;
        } else {
            int rc = self(self, tr[prev].rc, m + 1, r, pos, delta);
            tr[u].rc = rc;
        }
        return u;
    };

    auto pref_sum = [&](auto& self, int u, int l, int r, int qr) -> int {
        if (!u || l > qr) return 0;
        if (r <= qr) return tr[u].sum;
        int m = (l + r) >> 1;
        return self(self, tr[u].lc, l, m, qr) + self(self, tr[u].rc, m + 1, r, qr);
    };

    auto kth_pos = [&](auto& self, int u, int l, int r, int k) -> int {
        if (l == r) return l;
        int m = (l + r) >> 1;
        int lsum = tr[tr[u].lc].sum;
        if (k <= lsum) return self(self, tr[u].lc, l, m, k);
        return self(self, tr[u].rc, m + 1, r, k - lsum);
    };

    // Coordinate compress for O(1) last_seen lookup
    vector<long long> vals(a.begin() + 1, a.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    vector<int> last_pos(vals.size(), 0);

    vector<int> roots(n + 2, 0);
    for (int i = n; i >= 1; i--) {
        int cid = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
        int cur = roots[i + 1];
        if (last_pos[cid] != 0) {
            cur = upd(upd, cur, 1, n, last_pos[cid], -1);
        }
        cur = upd(upd, cur, 1, n, i, +1);
        last_pos[cid] = i;
        roots[i] = cur;
    }

    long long last_ans = 0;
    while (q--) {
        long long x, y, z;
        cin >> x >> y >> z;
        int l = (int)((x + last_ans) % n) + 1;
        int r = (int)((y + last_ans) % n) + 1;
        if (l > r) swap(l, r);
        int len = r - l + 1;
        int k = (int)((z + last_ans) % len) + 1;

        int cnt = pref_sum(pref_sum, roots[l], 1, n, r);
        if (cnt < k) {
            cout << -1 << '\n';
            last_ans = 0;
        } else {
            int idx = kth_pos(kth_pos, roots[l], 1, n, k);
            cout << a[idx] << '\n';
            last_ans = a[idx];
        }
    }
    return 0;
}
"""

BRUTE_115 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<long long> vals(a.begin() + 1, a.end());
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    vector<int> ca(n + 1);
    for (int i = 1; i <= n; i++) {
        ca[i] = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
    }
    vector<int> seen(vals.size(), 0);
    int timer = 0;

    long long last_ans = 0;
    while (q--) {
        long long x, y, z;
        cin >> x >> y >> z;
        int l = (int)((x + last_ans) % n) + 1;
        int r = (int)((y + last_ans) % n) + 1;
        if (l > r) swap(l, r);
        int len = r - l + 1;
        int k = (int)((z + last_ans) % len) + 1;

        ++timer;
        int cnt = 0;
        long long res = -1;
        for (int i = l; i <= r; i++) {
            int c = ca[i];
            if (seen[c] != timer) {
                seen[c] = timer;
                if (++cnt == k) {
                    res = a[i];
                    break;
                }
            }
        }
        cout << res << '\n';
        last_ans = (res == -1 ? 0 : res);
    }
    return 0;
}
"""


def gen_115(n: int, q: int, alpha: int, seed: int) -> str:
    r = random.Random(seed)
    palette = [r.randint(1, 10**9) for _ in range(alpha)]
    a = [r.choice(palette) for _ in range(n)]
    lines = [f"{n} {q}", " ".join(map(str, a))]
    for _ in range(q):
        rx = r.randrange(n)
        ry = r.randrange(n)
        rz = r.randrange(n)
        lines.append(f"{rx} {ry} {rz}")
    return "\n".join(lines) + "\n"


def make_specs() -> list[ProblemSpec]:
    return [
        ProblemSpec(
            pid="synth_111",
            title="Dynamic Canyon Span",
            rating=2900,
            tags=["trees", "data structures", "segment tree", "dfs and similar"],
            notes="CEOI 2019 Dynamic Diameter: full Euler tour + 5-value (max, min, d-2d, -2d+d, d-2d+d) lazy segment tree.",
            story_and_spec="""A canyon trail system consists of N camps numbered 1 to N connected by N - 1 bidirectional trails forming a tree. Trail i (1 <= i <= N - 1) connects camps U_i and V_i and initially has length W_i.

Process Q trail updates. Each update gives a trail index eid (1 <= eid <= N - 1) and a new positive length nw, permanently replacing trail eid's length with nw. After each update, print the diameter of the tree — the maximum shortest-path distance between any pair of camps.

Input
The first line contains N and Q.
Each of the next N - 1 lines contains U_i, V_i, and W_i.
Each of the next Q lines contains eid and nw.

Output
After each update print the updated tree diameter on its own line.

Constraints
2 <= N <= 120000
1 <= Q <= 120000
1 <= W_i, nw <= 10^9""",
            samples=[
                "5 4\n1 2 5\n2 3 6\n2 4 4\n4 5 7\n1 10\n4 1\n2 20\n3 2\n",
                "3 3\n1 2 8\n2 3 9\n1 3\n2 1\n1 15\n",
            ],
            small_inputs=[gen_111(2000, 2000, s, chain=(s == 1)) for s in range(1, 5)],
            large_inputs=[
                gen_111(120000, 120000, 101, chain=True),
                gen_111(120000, 120000, 102, chain=False),
                gen_111(110000, 120000, 103, chain=True),
                gen_111(120000, 115000, 104, chain=False),
            ],
            fast_cpp=FAST_111,
            brute_cpp=BRUTE_111,
        ),
        ProblemSpec(
            pid="synth_112",
            title="Bitwise Conduit Gates",
            rating=2900,
            tags=["data structures", "segtree beats", "bitmasks"],
            notes="Bitwise AND/OR Segment Tree Beats maintaining both range maximum and exact range sum via uniform-bit potential.",
            story_and_spec="""An array of N 30-bit registers holds non-negative integers A_1, ..., A_N.

Process Q operations of three types:
- 1 L R X : for each i in [L, R], set A_i <- A_i & X (bitwise AND).
- 2 L R X : for each i in [L, R], set A_i <- A_i | X (bitwise OR).
- 3 L R   : print two space-separated integers: max_{L <= i <= R} A_i and sum_{i=L}^R A_i.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines describes an operation in one of the formats above.

Output
For each type-3 operation print the range maximum and range sum separated by a space.

Constraints
1 <= N, Q <= 150000
0 <= A_i, X < 2^30
1 <= L <= R <= N""",
            samples=[
                "5 6\n7 12 5 9 15\n1 1 4 13\n3 1 5\n2 2 5 6\n3 2 4\n1 1 5 8\n3 1 5\n",
                "4 4\n0 0 0 0\n2 1 3 10\n2 2 4 5\n3 1 4\n1 1 4 12\n",
            ],
            small_inputs=[gen_112(2500, 2500, s, wide=True) for s in range(1, 5)],
            large_inputs=[
                gen_112(150000, 150000, 201, wide=True),
                gen_112(150000, 150000, 202, wide=False),
                gen_112(140000, 150000, 203, wide=True),
                gen_112(150000, 145000, 204, wide=True),
            ],
            fast_cpp=FAST_112,
            brute_cpp=BRUTE_112,
        ),
        ProblemSpec(
            pid="synth_113",
            title="Ordinal Trail Census",
            rating=2800,
            tags=["trees", "divide and conquer", "centroid decomposition", "binary search", "two pointers"],
            notes="K-th smallest tree path length and sum of K smallest path lengths via pre-sorted Centroid Decomposition + binary search.",
            story_and_spec="""A forest reserve has N clearings numbered 1 to N connected by N - 1 bidirectional trails of positive lengths forming a tree.

Consider the N * (N - 1) / 2 unordered pairs of distinct clearings {u, v} (1 <= u < v <= N) and sort their shortest-path distances in non-decreasing order: d_1 <= d_2 <= ... <= d_{N*(N-1)/2}.

Given K, find d_K (the K-th smallest distance) and the sum (d_1 + d_2 + ... + d_K) modulo 998244353.

Input
The first line contains N and K.
Each of the next N - 1 lines contains u, v, and w describing a trail of length w.

Output
Print two space-separated integers: d_K and (d_1 + ... + d_K) mod 998244353.

Constraints
2 <= N <= 75000
1 <= K <= N * (N - 1) / 2
1 <= w <= 10^8""",
            samples=[
                "4 4\n1 2 3\n2 3 4\n2 4 5\n",
                "5 7\n1 2 2\n1 3 2\n1 4 2\n1 5 2\n",
            ],
            small_inputs=[
                gen_113(550, 1, high_k=True),
                gen_113(600, 2, high_k=False),
                gen_113(500, 3, high_k=True),
                "2 1\n1 2 100000000\n",
            ],
            large_inputs=[
                gen_113(75000, 301, high_k=True),
                gen_113(75000, 302, high_k=False),
                gen_113(70000, 303, high_k=True),
                gen_113(75000, 304, high_k=True),
            ],
            fast_cpp=FAST_113,
            brute_cpp=BRUTE_113,
        ),
        ProblemSpec(
            pid="synth_114",
            title="Ephemeral Profit Tariffs",
            rating=2800,
            tags=["geometry", "convex hull", "data structures", "divide and conquer"],
            notes="Offline dynamic convex hull trick with insertions and deletions via Time Segment Tree of Upper Hulls.",
            story_and_spec="""A commodities exchange opens and closes linear tariff contracts over Q steps. Initially no contracts are active.

At each step, one of three events occurs:
- 1 a b : open a new tariff contract with slope a and intercept b. Newly opened contracts are numbered 1, 2, 3, ... in the order they appear.
- 2 id  : close the currently active contract with identifier id (guaranteed to be open and not yet closed).
- 3 x   : evaluate max (a * x + b) over all currently active contracts. If no contract is active, print EMPTY.

Input
The first line contains Q.
Each of the next Q lines describes an event in one of the three formats above.

Output
For each type-3 event print the maximum value (or EMPTY) on its own line.

Constraints
1 <= Q <= 220000
-10^6 <= a, x <= 10^6
-10^12 <= b <= 10^12""",
            samples=[
                "8\n3 5\n1 2 3\n1 -1 15\n3 2\n3 6\n2 1\n3 6\n2 2\n",
                "6\n1 0 -10\n3 100\n2 1\n3 100\n1 5 0\n3 -3\n",
            ],
            small_inputs=[gen_114(2500, s, bulk_lines=True) for s in range(1, 5)],
            large_inputs=[
                gen_114(220000, 401, bulk_lines=True),
                gen_114(200000, 402, bulk_lines=False),
                gen_114(210000, 403, bulk_lines=True),
                gen_114(220000, 404, bulk_lines=True),
            ],
            fast_cpp=FAST_114,
            brute_cpp=BRUTE_114,
        ),
        ProblemSpec(
            pid="synth_115",
            title="Online Relic Catalogue",
            rating=2700,
            tags=["data structures", "persistent segment tree"],
            notes="Strictly online K-th first-occurrence distinct element in subarray [L, R] via suffix Persistent Segment Tree.",
            story_and_spec="""A museum corridor displays N relics in order, with catalog codes A_1, ..., A_N.

You must answer Q online queries. Maintain an integer `last_ans`, initially 0. Each query provides non-negative integers x, y, z. Decode L, R, K as follows:
l_raw = (x + last_ans) mod N + 1,
r_raw = (y + last_ans) mod N + 1,
L = min(l_raw, r_raw),  R = max(l_raw, r_raw),
K = (z + last_ans) mod (R - L + 1) + 1.

Scan the subarray A_L, A_{L+1}, ..., A_R from left to right and retain only the FIRST appearance of each distinct catalog code within [L, R], producing a list U_1, U_2, ..., U_m of distinct codes ordered by their leftmost occurrence in [L, R].

If m < K, print -1 and set `last_ans <- 0`. Otherwise print U_K and set `last_ans <- U_K`.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains x, y, and z.

Output
For each query print the answer on its own line.

Constraints
1 <= N, Q <= 180000
1 <= A_i <= 10^9
0 <= x, y, z < N""",
            samples=[
                "6 5\n4 2 4 7 2 9\n0 5 0\n0 5 2\n1 4 3\n2 3 1\n0 5 3\n",
                "4 3\n5 5 5 5\n0 3 0\n0 3 1\n1 2 0\n",
            ],
            small_inputs=[
                gen_115(2400, 2400, 40, 1),
                gen_115(2500, 2500, 800, 2),
                gen_115(2200, 2400, 10, 3),
                gen_115(2500, 2500, 2500, 4),
            ],
            large_inputs=[
                gen_115(180000, 180000, 180000, 501),
                gen_115(180000, 180000, 20000, 502),
                gen_115(170000, 180000, 300, 503),
                gen_115(180000, 180000, 100000, 504),
            ],
            fast_cpp=FAST_115,
            brute_cpp=BRUTE_115,
        ),
    ]


if __name__ == "__main__":
    build_and_verify_batch(make_specs())
