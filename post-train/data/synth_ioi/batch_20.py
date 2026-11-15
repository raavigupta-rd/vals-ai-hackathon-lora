"""Batch 20: synth_96 .. synth_100 (Grandmaster Wave IV: CF 2700-2900)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from builder import ProblemSpec, build_and_verify_batch


# ---------------------------------------------------------------------------
# synth_96 — 3D Strict Partial-Order Max-Weight Chain & Count (CDQ + BIT)
# ---------------------------------------------------------------------------
FAST_96 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

struct Val {
    long long v;
    int ways;
};

static inline Val merge_val(const Val& a, const Val& b) {
    if (a.v > b.v) return a;
    if (b.v > a.v) return b;
    int w = a.ways + b.ways;
    if (w >= MOD) w -= MOD;
    return {a.v, w};
}

struct Pt {
    int x, y, z;
    long long w;
    int id;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<Pt> pts(n);
    vector<int> zs(n);
    for (int i = 0; i < n; i++) {
        cin >> pts[i].x >> pts[i].y >> pts[i].z >> pts[i].w;
        pts[i].id = i;
        zs[i] = pts[i].z;
    }
    sort(zs.begin(), zs.end());
    zs.erase(unique(zs.begin(), zs.end()), zs.end());
    int mz = (int)zs.size();
    for (int i = 0; i < n; i++) {
        pts[i].z = (int)(lower_bound(zs.begin(), zs.end(), pts[i].z) - zs.begin()) + 1;
    }

    sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    });

    // Group strictly by distinct X so left groups always have X_left < X_right
    vector<pair<int, int>> groups;
    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && pts[j].x == pts[i].x) j++;
        groups.push_back({i, j});
        i = j;
    }

    vector<Val> dp(n, {0, 0});
    for (int i = 0; i < n; i++) {
        dp[pts[i].id] = {pts[i].w, 1};
    }

    vector<Val> bit(mz + 2, {0, 0});
    auto bit_upd = [&](int idx, const Val& val) {
        for (; idx <= mz; idx += idx & -idx) bit[idx] = merge_val(bit[idx], val);
    };
    auto bit_qry = [&](int idx) -> Val {
        Val res{0, 0};
        for (; idx > 0; idx -= idx & -idx) res = merge_val(res, bit[idx]);
        return res;
    };
    auto bit_clr = [&](int idx) {
        for (; idx <= mz; idx += idx & -idx) {
            if (bit[idx].v == 0 && bit[idx].ways == 0) break;
            bit[idx] = {0, 0};
        }
    };

    vector<Pt> left_buf, right_buf;
    left_buf.reserve(n);
    right_buf.reserve(n);

    auto cdq = [&](auto& self, int gl, int gr) -> void {
        if (gl >= gr) return;
        int gm = (gl + gr) >> 1;
        self(self, gl, gm);

        int l_start = groups[gl].first, l_end = groups[gm].second;
        int r_start = groups[gm + 1].first, r_end = groups[gr].second;
        left_buf.assign(pts.begin() + l_start, pts.begin() + l_end);
        right_buf.assign(pts.begin() + r_start, pts.begin() + r_end);

        auto cmp_y = [](const Pt& a, const Pt& b) {
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        };
        sort(left_buf.begin(), left_buf.end(), cmp_y);
        sort(right_buf.begin(), right_buf.end(), cmp_y);

        int p = 0, lsz = (int)left_buf.size();
        for (const auto& rq : right_buf) {
            while (p < lsz && left_buf[p].y < rq.y) {
                bit_upd(left_buf[p].z, dp[left_buf[p].id]);
                p++;
            }
            Val best = bit_qry(rq.z - 1);
            if (best.ways > 0) {
                Val cand{best.v + rq.w, best.ways};
                dp[rq.id] = merge_val(dp[rq.id], cand);
            }
        }
        for (int i = 0; i < p; i++) bit_clr(left_buf[i].z);

        self(self, gm + 1, gr);
    };

    if (!groups.empty()) cdq(cdq, 0, (int)groups.size() - 1);

    Val ans{0, 0};
    for (int i = 0; i < n; i++) ans = merge_val(ans, dp[i]);
    cout << ans.v << ' ' << ans.ways << '\n';
    return 0;
}
"""

BRUTE_96 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

struct Pt {
    int x, y, z;
    long long w;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<Pt> p(n);
    for (int i = 0; i < n; i++) cin >> p[i].x >> p[i].y >> p[i].z >> p[i].w;
    sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    });
    vector<long long> best_v(n);
    vector<int> ways(n);
    long long ans_v = 0;
    int ans_w = 0;
    for (int i = 0; i < n; i++) {
        long long pv = 0;
        int pw = 0;
        for (int j = 0; j < i; j++) {
            if (p[j].x < p[i].x && p[j].y < p[i].y && p[j].z < p[i].z) {
                if (best_v[j] > pv) {
                    pv = best_v[j];
                    pw = ways[j];
                } else if (best_v[j] == pv) {
                    pw += ways[j];
                    if (pw >= MOD) pw -= MOD;
                }
            }
        }
        if (pw == 0) {
            best_v[i] = p[i].w;
            ways[i] = 1;
        } else {
            best_v[i] = pv + p[i].w;
            ways[i] = pw;
        }
        if (best_v[i] > ans_v) {
            ans_v = best_v[i];
            ans_w = ways[i];
        } else if (best_v[i] == ans_v) {
            ans_w = (ans_w + ways[i]) % MOD;
        }
    }
    cout << ans_v << ' ' << ans_w << '\n';
    return 0;
}
"""


def gen_96(n: int, coord_lim: int, seed: int) -> str:
    r = random.Random(seed)
    lines = [str(n)]
    # Mix layered chains and random points to create ties in optimal chain weight
    layers = max(3, int(n ** 0.5))
    for i in range(n):
        lay = r.randint(1, layers)
        span = max(2, coord_lim // layers)
        lo = (lay - 1) * span + 1
        hi = lay * span
        x = r.randint(lo, hi)
        y = r.randint(lo, hi)
        z = r.randint(lo, hi)
        w = r.randint(1, 20) if seed % 2 == 0 else r.randint(1, 10**9)
        lines.append(f"{x} {y} {z} {w}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_97 — Max Subarray Minimum of Width W in [L, R] (Persistent SegTree)
# ---------------------------------------------------------------------------
FAST_97 = r"""
#include <bits/stdc++.h>
using namespace std;

struct Info {
    int len, pref, suff, best;
};

static inline Info merge_info(const Info& a, const Info& b) {
    if (a.len == 0) return b;
    if (b.len == 0) return a;
    Info res;
    res.len = a.len + b.len;
    res.pref = (a.pref == a.len) ? (a.len + b.pref) : a.pref;
    res.suff = (b.suff == b.len) ? (b.len + a.suff) : b.suff;
    res.best = max({a.best, b.best, a.suff + b.pref});
    return res;
}

struct Node {
    int lc, rc;
    int pref, suff, best;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<pair<int, int>> ord(n);
    for (int i = 1; i <= n; i++) {
        int h;
        cin >> h;
        ord[i - 1] = {h, i};
    }
    sort(ord.begin(), ord.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    vector<Node> tr;
    tr.reserve(n * 20 + 4 * n + 10);
    tr.push_back({0, 0, 0, 0, 0});

    auto build = [&](auto& self, int l, int r) -> int {
        int u = (int)tr.size();
        tr.push_back({0, 0, 0, 0, 0});
        if (l == r) return u;
        int m = (l + r) >> 1;
        int lc = self(self, l, m);
        int rc = self(self, m + 1, r);
        tr[u].lc = lc;
        tr[u].rc = rc;
        return u;
    };

    auto update = [&](auto& self, int prev, int l, int r, int pos) -> int {
        int u = (int)tr.size();
        tr.push_back(tr[prev]);
        if (l == r) {
            tr[u].pref = tr[u].suff = tr[u].best = 1;
            return u;
        }
        int m = (l + r) >> 1;
        if (pos <= m) {
            int lc = self(self, tr[prev].lc, l, m, pos);
            tr[u].lc = lc;
        } else {
            int rc = self(self, tr[prev].rc, m + 1, r, pos);
            tr[u].rc = rc;
        }
        int lc = tr[u].lc, rc = tr[u].rc;
        int llen = m - l + 1, rlen = r - m;
        tr[u].pref = (tr[lc].pref == llen) ? (llen + tr[rc].pref) : tr[lc].pref;
        tr[u].suff = (tr[rc].suff == rlen) ? (rlen + tr[lc].suff) : tr[rc].suff;
        tr[u].best = max({tr[lc].best, tr[rc].best, tr[lc].suff + tr[rc].pref});
        return u;
    };

    auto query = [&](auto& self, int u, int l, int r, int ql, int qr) -> Info {
        if (ql <= l && r <= qr) {
            return {r - l + 1, tr[u].pref, tr[u].suff, tr[u].best};
        }
        int m = (l + r) >> 1;
        if (qr <= m) return self(self, tr[u].lc, l, m, ql, qr);
        if (ql > m) return self(self, tr[u].rc, m + 1, r, ql, qr);
        return merge_info(self(self, tr[u].lc, l, m, ql, qr),
                          self(self, tr[u].rc, m + 1, r, ql, qr));
    };

    vector<int> roots(n + 1, 0);
    roots[0] = build(build, 1, n);
    for (int i = 1; i <= n; i++) {
        roots[i] = update(update, roots[i - 1], 1, n, ord[i - 1].second);
    }

    while (q--) {
        int l, r, w;
        cin >> l >> r >> w;
        int lo = 1, hi = n, ans_idx = n;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (query(query, roots[mid], 1, n, l, r).best >= w) {
                ans_idx = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        cout << ord[ans_idx - 1].first << '\n';
    }
    return 0;
}
"""

BRUTE_97 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> h(n + 1);
    for (int i = 1; i <= n; i++) cin >> h[i];
    vector<int> dq(n + 2);
    while (q--) {
        int l, r, w;
        cin >> l >> r >> w;
        int head = 0, tail = 0;
        int best = 0;
        for (int i = l; i <= r; i++) {
            while (head < tail && h[dq[tail - 1]] >= h[i]) tail--;
            dq[tail++] = i;
            if (dq[head] <= i - w) head++;
            if (i >= l + w - 1) best = max(best, h[dq[head]]);
        }
        cout << best << '\n';
    }
    return 0;
}
"""


def gen_97(n: int, q: int, seed: int, wide_queries: bool = False) -> str:
    r = random.Random(seed)
    h = [r.randint(1, 10**9) for _ in range(n)]
    lines = [f"{n} {q}", " ".join(map(str, h))]
    for _ in range(q):
        if wide_queries and r.random() < 0.85:
            ql = r.randint(1, max(1, n // 10))
            qr = r.randint(max(ql, n - n // 10), n)
        else:
            a = r.randint(1, n)
            b = r.randint(1, n)
            ql, qr = min(a, b), max(a, b)
        w = r.randint(1, qr - ql + 1)
        lines.append(f"{ql} {qr} {w}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_98 — High-Order Linear Recurrence at K <= 10^18 (Bostan-Mori + NTT)
# ---------------------------------------------------------------------------
FAST_98 = r"""
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

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int d;
    long long k;
    if (!(cin >> d >> k)) return 0;
    vector<int> a(d), c(d + 1, 0);
    for (int i = 0; i < d; i++) {
        cin >> a[i];
        a[i] %= MOD;
    }
    for (int j = 1; j <= d; j++) {
        cin >> c[j];
        c[j] %= MOD;
    }
    if (k < d) {
        cout << a[k] << '\n';
        return 0;
    }
    // Q(x) = 1 - sum_{j=1}^d c_j x^j
    vector<int> Q(d + 1, 0);
    Q[0] = 1;
    for (int j = 1; j <= d; j++) Q[j] = (MOD - c[j]) % MOD;
    // P(x) = (A(x) * Q(x)) mod x^d
    vector<int> P = poly_mul(a, Q);
    P.resize(d);

    while (k > 0) {
        vector<int> Qneg = Q;
        for (int i = 1; i < (int)Qneg.size(); i += 2) {
            if (Qneg[i]) Qneg[i] = MOD - Qneg[i];
        }
        vector<int> U = poly_mul(P, Qneg);
        vector<int> V = poly_mul(Q, Qneg);
        int parity = (int)(k & 1LL);
        vector<int> nP, nQ;
        nP.reserve((U.size() + 1) / 2);
        nQ.reserve((V.size() + 1) / 2);
        for (int i = parity; i < (int)U.size(); i += 2) nP.push_back(U[i]);
        for (int i = 0; i < (int)V.size(); i += 2) nQ.push_back(V[i]);
        P.swap(nP);
        Q.swap(nQ);
        k >>= 1;
    }
    int ans = P.empty() ? 0 : (int)((long long)P[0] * mod_pow(Q[0], MOD - 2) % MOD);
    cout << ans << '\n';
    return 0;
}
"""

BRUTE_98 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int d;
    long long k;
    cin >> d >> k;
    vector<int> a(d), c(d);
    for (int i = 0; i < d; i++) cin >> a[i];
    for (int j = 0; j < d; j++) cin >> c[j];
    if (k < d) {
        cout << a[k] % MOD << '\n';
        return 0;
    }
    // Rolling ring buffer O(D) memory, O(K * D) time
    vector<int> buf = a;
    int head = 0;
    for (long long step = d; step <= k; step++) {
        long long acc = 0;
        for (int j = 1; j <= d; j++) {
            int idx = head - j;
            if (idx < 0) idx += d;
            acc += (long long)c[j - 1] * buf[idx] % MOD;
        }
        buf[head] = (int)(acc % MOD);
        head++;
        if (head == d) head = 0;
    }
    int last = (head == 0 ? d - 1 : head - 1);
    cout << buf[last] << '\n';
    return 0;
}
"""


def gen_98(d: int, k: int, seed: int) -> str:
    r = random.Random(seed)
    mod = 998244353
    a = [r.randint(0, mod - 1) for _ in range(d)]
    c = [r.randint(0, mod - 1) for _ in range(d)]
    return f"{d} {k}\n" + " ".join(map(str, a)) + "\n" + " ".join(map(str, c)) + "\n"


# ---------------------------------------------------------------------------
# synth_99 — Dynamic Weighted Distance Sum on Tree (Centroid Decomposition)
# ---------------------------------------------------------------------------
FAST_99 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    static const int LOG = 19;
    vector<array<int, LOG>> anc(n + 1);
    vector<array<long long, LOG>> d_anc(n + 1);
    vector<int> len_anc(n + 1, 0);

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), par_tmp(n + 1, 0), nodes;
    nodes.reserve(n);

    auto build_cd = [&](auto& self, int entry) -> void {
        nodes.clear();
        par_tmp[entry] = 0;
        nodes.push_back(entry);
        for (size_t i = 0; i < nodes.size(); i++) {
            int u = nodes[i];
            sz[u] = 1;
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_tmp[u]) {
                    par_tmp[v] = u;
                    nodes.push_back(v);
                }
            }
        }
        int tot = (int)nodes.size();
        for (int i = tot - 1; i >= 0; i--) {
            int u = nodes[i];
            if (par_tmp[u]) sz[par_tmp[u]] += sz[u];
        }
        int cent = entry;
        for (int u : nodes) {
            int mx = tot - sz[u];
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_tmp[u]) mx = max(mx, sz[v]);
            }
            if (mx * 2 <= tot) {
                cent = u;
                break;
            }
        }

        // Iterative BFS from cent to record distance to cent
        nodes.clear();
        par_tmp[cent] = 0;
        int lvl_c = len_anc[cent];
        anc[cent][lvl_c] = cent;
        d_anc[cent][lvl_c] = 0;
        len_anc[cent]++;
        nodes.push_back(cent);
        for (size_t i = 0; i < nodes.size(); i++) {
            int u = nodes[i];
            long long du = d_anc[u][len_anc[u] - 1];
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_tmp[u]) {
                    par_tmp[v] = u;
                    int lv = len_anc[v];
                    anc[v][lv] = cent;
                    d_anc[v][lv] = du + w;
                    len_anc[v]++;
                    nodes.push_back(v);
                }
            }
        }

        dead[cent] = 1;
        for (auto [v, w] : adj[cent]) {
            if (!dead[v]) self(self, v);
        }
    };

    build_cd(build_cd, 1);

    vector<long long> sum_a(n + 1, 0), sum_ad(n + 1, 0), sub_ad(n + 1, 0);

    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u;
            long long delta;
            cin >> u >> delta;
            int k = len_anc[u];
            for (int l = 0; l < k; l++) {
                int c = anc[u][l];
                long long d = d_anc[u][l];
                sum_a[c] += delta;
                sum_ad[c] += delta * d;
                if (l + 1 < k) {
                    int ch = anc[u][l + 1];
                    sub_ad[ch] += delta * d;
                }
            }
        } else {
            int u;
            cin >> u;
            long long ans = 0;
            int k = len_anc[u];
            for (int l = 0; l < k; l++) {
                int c = anc[u][l];
                long long d = d_anc[u][l];
                if (l == k - 1) {
                    ans += sum_ad[c];
                } else {
                    int ch = anc[u][l + 1];
                    ans += (sum_ad[c] - sub_ad[ch]) + (sum_a[c] - sum_a[ch]) * d;
                }
            }
            cout << ans << '\n';
        }
    }
    return 0;
}
"""

BRUTE_99 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    vector<long long> a(n + 1, 0), dist(n + 1, 0);
    vector<int> par(n + 1, 0), que(n + 1);
    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u;
            long long d;
            cin >> u >> d;
            a[u] += d;
        } else {
            int src;
            cin >> src;
            int head = 0, tail = 0;
            que[tail++] = src;
            par[src] = 0;
            dist[src] = 0;
            long long ans = 0;
            while (head < tail) {
                int u = que[head++];
                ans += a[u] * dist[u];
                for (auto [v, w] : adj[u]) {
                    if (v != par[u]) {
                        par[v] = u;
                        dist[v] = dist[u] + w;
                        que[tail++] = v;
                    }
                }
            }
            cout << ans << '\n';
        }
    }
    return 0;
}
"""


def gen_99(n: int, q: int, seed: int, chain: bool = False) -> str:
    r = random.Random(seed)
    lines = [f"{n} {q}"]
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    for i in range(1, n):
        p = perm[i - 1] if chain else perm[r.randint(max(0, i - 6), i - 1)]
        w = r.randint(1, 1000)
        lines.append(f"{p} {perm[i]} {w}")
    # Start with several updates so initial queries are non-trivial
    for idx in range(q):
        if idx < min(q // 3, 2000) or r.random() < 0.45:
            u = r.randint(1, n)
            delta = r.randint(-1000, 1000)
            lines.append(f"1 {u} {delta}")
        else:
            u = r.randint(1, n)
            lines.append(f"2 {u}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_100 — Historic Peak Ledger (Jiry (add, max) Tropical Semigroup SegTree)
# ---------------------------------------------------------------------------
FAST_100 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 60);

struct Tag {
    long long a, b; // f(x) = max(x + a, b)
};

static inline long long add_clamped(long long x, long long y) {
    if (x <= NEG_INF / 2 || y <= NEG_INF / 2) return NEG_INF;
    long long s = x + y;
    if (s <= NEG_INF) return NEG_INF;
    return s;
}

static inline long long eval_tag(const Tag& t, long long x) {
    return max(add_clamped(x, t.a), t.b);
}

// g(f(x)) = max(max(x + f.a, f.b) + g.a, g.b)
static inline Tag compose_tag(const Tag& g, const Tag& f) {
    return {add_clamped(f.a, g.a), max(add_clamped(f.b, g.a), g.b)};
}

static inline Tag max_tag(const Tag& f, const Tag& g) {
    return {max(f.a, g.a), max(f.b, g.b)};
}

struct SegNode {
    long long max_a, max_b;
    Tag cur, hist;
    bool dirty;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> init_a(n + 1);
    for (int i = 1; i <= n; i++) cin >> init_a[i];

    const Tag ID{0, NEG_INF};
    vector<SegNode> tr(4 * n + 5, {NEG_INF, NEG_INF, ID, ID, false});

    auto apply_window = [&](int u, const Tag& w_cur, const Tag& w_hist) {
        long long old_a = tr[u].max_a;
        tr[u].max_a = eval_tag(w_cur, old_a);
        tr[u].max_b = max(tr[u].max_b, eval_tag(w_hist, old_a));
        if (!tr[u].dirty) {
            tr[u].cur = w_cur;
            tr[u].hist = w_hist;
            tr[u].dirty = true;
        } else {
            tr[u].hist = max_tag(tr[u].hist, compose_tag(w_hist, tr[u].cur));
            tr[u].cur = compose_tag(w_cur, tr[u].cur);
        }
    };

    auto push = [&](int u) {
        if (!tr[u].dirty) return;
        apply_window(u << 1, tr[u].cur, tr[u].hist);
        apply_window(u << 1 | 1, tr[u].cur, tr[u].hist);
        tr[u].cur = ID;
        tr[u].hist = ID;
        tr[u].dirty = false;
    };

    auto pull = [&](int u) {
        tr[u].max_a = max(tr[u << 1].max_a, tr[u << 1 | 1].max_a);
        tr[u].max_b = max(tr[u << 1].max_b, tr[u << 1 | 1].max_b);
    };

    auto build = [&](auto& self, int u, int l, int r) -> void {
        if (l == r) {
            tr[u].max_a = init_a[l];
            tr[u].max_b = init_a[l];
            return;
        }
        int m = (l + r) >> 1;
        self(self, u << 1, l, m);
        self(self, u << 1 | 1, m + 1, r);
        pull(u);
    };

    auto update = [&](auto& self, int u, int l, int r, int ql, int qr, const Tag& tg) -> void {
        if (ql <= l && r <= qr) {
            apply_window(u, tg, tg);
            return;
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, tg);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, tg);
        pull(u);
    };

    auto query = [&](auto& self, int u, int l, int r, int ql, int qr) -> pair<long long, long long> {
        if (ql <= l && r <= qr) return {tr[u].max_a, tr[u].max_b};
        push(u);
        int m = (l + r) >> 1;
        pair<long long, long long> res{NEG_INF, NEG_INF};
        if (ql <= m) {
            auto lv = self(self, u << 1, l, m, ql, qr);
            res.first = max(res.first, lv.first);
            res.second = max(res.second, lv.second);
        }
        if (qr > m) {
            auto rv = self(self, u << 1 | 1, m + 1, r, ql, qr);
            res.first = max(res.first, rv.first);
            res.second = max(res.second, rv.second);
        }
        return res;
    };

    build(build, 1, 1, n);

    while (q--) {
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            long long x;
            cin >> x;
            update(update, 1, 1, n, l, r, {x, NEG_INF});
        } else if (tp == 2) {
            long long v;
            cin >> v;
            update(update, 1, 1, n, l, r, {0, v});
        } else if (tp == 3) {
            long long v;
            cin >> v;
            update(update, 1, 1, n, l, r, {NEG_INF, v});
        } else {
            auto [ma, mb] = query(query, 1, 1, n, l, r);
            cout << ma << ' ' << mb << '\n';
        }
    }
    return 0;
}
"""

BRUTE_100 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1), b(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        b[i] = a[i];
    }
    while (q--) {
        int tp, l, r;
        cin >> tp >> l >> r;
        if (tp == 1) {
            long long x;
            cin >> x;
            for (int i = l; i <= r; i++) {
                a[i] += x;
                if (a[i] > b[i]) b[i] = a[i];
            }
        } else if (tp == 2) {
            long long v;
            cin >> v;
            for (int i = l; i <= r; i++) {
                if (v > a[i]) a[i] = v;
                if (a[i] > b[i]) b[i] = a[i];
            }
        } else if (tp == 3) {
            long long v;
            cin >> v;
            for (int i = l; i <= r; i++) {
                a[i] = v;
                if (a[i] > b[i]) b[i] = a[i];
            }
        } else {
            long long ma = -(1LL << 60), mb = -(1LL << 60);
            for (int i = l; i <= r; i++) {
                ma = max(ma, a[i]);
                mb = max(mb, b[i]);
            }
            cout << ma << ' ' << mb << '\n';
        }
    }
    return 0;
}
"""


def gen_100(n: int, q: int, seed: int, wide: bool = False) -> str:
    r = random.Random(seed)
    a = [r.randint(-10**8, 10**8) for _ in range(n)]
    lines = [f"{n} {q}", " ".join(map(str, a))]
    for _ in range(q):
        tp = r.choices([1, 2, 3, 4], weights=[35, 20, 15, 30])[0]
        if wide and r.random() < 0.8:
            ql = r.randint(1, max(1, n // 10))
            qr = r.randint(max(ql, n - n // 10), n)
        else:
            x = r.randint(1, n)
            y = r.randint(1, n)
            ql, qr = min(x, y), max(x, y)
        if tp == 1:
            lines.append(f"1 {ql} {qr} {r.randint(-10**8, 10**8)}")
        elif tp == 2:
            lines.append(f"2 {ql} {qr} {r.randint(-10**8, 10**8)}")
        elif tp == 3:
            lines.append(f"3 {ql} {qr} {r.randint(-10**8, 10**8)}")
        else:
            lines.append(f"4 {ql} {qr}")
    return "\n".join(lines) + "\n"


def make_specs() -> list[ProblemSpec]:
    return [
        ProblemSpec(
            pid="synth_96",
            title="Three Axis Dominance",
            rating=2700,
            tags=["divide and conquer", "data structures", "dp", "cdq"],
            notes="3D strict partial-order max-weight chain and count via X-grouped CDQ divide-and-conquer + Fenwick tree.",
            story_and_spec="""An observatory maps N crystal nodes in 3D space. Node i has integer coordinates (X_i, Y_i, Z_i) and resonance weight W_i.

A sequence of nodes i_1, i_2, ..., i_k (k >= 1) forms a valid resonance chain if every step strictly increases all three coordinates:
X_{i_t} < X_{i_{t+1}}, Y_{i_t} < Y_{i_{t+1}}, and Z_{i_t} < Z_{i_{t+1}} for all 1 <= t < k.

Find the maximum possible total weight W_{i_1} + ... + W_{i_k} of a valid resonance chain, and count how many distinct index sequences (i_1, ..., i_k) achieve this maximum total weight, modulo 998244353.

Input
The first line contains N.
Each of the next N lines contains X_i, Y_i, Z_i, and W_i.

Output
Print two space-separated integers: the maximum chain weight and the number of optimal chains modulo 998244353.

Constraints
1 <= N <= 100000
1 <= X_i, Y_i, Z_i <= 10^9
1 <= W_i <= 10^9""",
            samples=[
                "5\n1 1 1 5\n2 2 2 4\n2 3 2 4\n3 4 4 6\n2 1 5 20\n",
                "3\n5 5 5 10\n5 6 7 12\n1 2 3 1\n",
            ],
            small_inputs=[gen_96(2600, 3000, s) for s in range(1, 5)],
            large_inputs=[
                gen_96(100000, 10**9, 101),
                gen_96(100000, 500, 102),
                gen_96(95000, 10**9, 103),
                gen_96(100000, 5000, 104),
            ],
            fast_cpp=FAST_96,
            brute_cpp=BRUTE_96,
        ),
        ProblemSpec(
            pid="synth_97",
            title="Banner on the Palisade",
            rating=2800,
            tags=["data structures", "persistent segment tree", "binary search"],
            notes="CF 484E style: binary search over persistent max-contiguous-ones segment tree.",
            story_and_spec="""A wooden palisade consists of N vertical planks placed side by side, numbered 1 to N. Plank i has width 1 and height H_i.

You must answer Q queries. Each query specifies an interval of planks [L, R] and a banner width W (1 <= W <= R - L + 1). You want to hang a solid rectangular banner of width W and height h wholly inside planks [L, R] with its base resting at height 0, so it covers W consecutive planks s, s+1, ..., s+W-1 (L <= s <= R - W + 1) and requires H_i >= h for every covered plank i.

For each query, print the maximum possible integer height h.

Input
The first line contains N and Q.
The second line contains H_1 ... H_N.
Each of the next Q lines contains L, R, and W.

Output
For each query print the maximum banner height on its own line.

Constraints
1 <= N, Q <= 100000
1 <= H_i <= 10^9
1 <= L <= R <= N
1 <= W <= R - L + 1""",
            samples=[
                "6 4\n4 8 6 7 3 9\n1 6 3\n2 5 2\n1 4 4\n5 6 1\n",
                "4 2\n5 5 5 5\n1 4 3\n2 3 2\n",
            ],
            small_inputs=[gen_97(2400, 2400, s, wide_queries=True) for s in range(1, 5)],
            large_inputs=[
                gen_97(100000, 100000, 201, wide_queries=True),
                gen_97(100000, 100000, 202, wide_queries=False),
                gen_97(95000, 100000, 203, wide_queries=True),
                gen_97(100000, 95000, 204, wide_queries=True),
            ],
            fast_cpp=FAST_97,
            brute_cpp=BRUTE_97,
        ),
        ProblemSpec(
            pid="synth_98",
            title="Harmonic Beacon Recurrence",
            rating=2800,
            tags=["fft", "ntt", "math", "polynomials"],
            notes="Bostan-Mori rational generating function P(x)/Q(x) evaluation with NTT in O(D log D log K).",
            story_and_spec="""A deep-space harmonic beacon emits pulses governed by a linear recurrence of order D. You are given the initial pulse amplitudes A_0, A_1, ..., A_{D-1} and recurrence coefficients C_1, C_2, ..., C_D. For every integer n >= D,
A_n = (C_1 * A_{n-1} + C_2 * A_{n-2} + ... + C_D * A_{n-D}) mod 998244353.

Compute A_K mod 998244353.

Input
The first line contains D and K.
The second line contains A_0 ... A_{D-1}.
The third line contains C_1 ... C_D.

Output
Print A_K modulo 998244353.

Constraints
1 <= D <= 8000
0 <= K <= 10^18
0 <= A_i, C_j < 998244353""",
            samples=[
                "2 10\n0 1\n1 1\n",
                "3 7\n1 2 4\n2 3 5\n",
            ],
            small_inputs=[
                gen_98(350, 24000, 1),
                gen_98(400, 25000, 2),
                gen_98(250, 3, 3),
                gen_98(1, 20000, 4),
            ],
            large_inputs=[
                gen_98(8000, 10**18, 301),
                gen_98(7500, 987654321987654321, 302),
                gen_98(8000, 500000000000000000, 303),
                gen_98(6000, 10**18 - 7, 304),
            ],
            fast_cpp=FAST_98,
            brute_cpp=BRUTE_98,
        ),
        ProblemSpec(
            pid="synth_99",
            title="Emergency Beacon Network",
            rating=2800,
            tags=["trees", "divide and conquer", "centroid decomposition"],
            notes="Dynamic weighted vertex updates and all-pairs distance sum queries via Centroid Decomposition.",
            story_and_spec="""A mountain rescue network connects N stations with N - 1 bidirectional trails. Trail i connects U_i and V_i with length W_i. Initially every station u has demand weight A_u = 0.

Process Q events of two types:
- 1 u delta : change A_u by adding delta (note that A_u may become negative).
- 2 u       : compute sum_{v=1}^N A_v * dist(u, v), where dist(u, v) is the shortest trail distance between u and v.

Input
The first line contains N and Q.
Each of the next N - 1 lines contains U_i, V_i, and W_i.
Each of the next Q lines describes an event in one of the two formats above.

Output
For each type-2 event print the weighted distance sum on its own line.

Constraints
1 <= N <= 100000
1 <= Q <= 120000
1 <= W_i <= 1000
-1000 <= delta <= 1000""",
            samples=[
                "5 6\n1 2 3\n2 3 4\n2 4 2\n4 5 5\n1 1 2\n1 5 3\n2 2\n1 3 -1\n2 4\n2 1\n",
                "3 4\n1 2 10\n2 3 20\n1 1 5\n2 3\n1 3 -2\n2 1\n",
            ],
            small_inputs=[gen_99(2400, 2400, s, chain=(s == 1)) for s in range(1, 5)],
            large_inputs=[
                gen_99(100000, 120000, 401, chain=True),
                gen_99(100000, 120000, 402, chain=False),
                gen_99(95000, 120000, 403, chain=False),
                gen_99(100000, 115000, 404, chain=True),
            ],
            fast_cpp=FAST_99,
            brute_cpp=BRUTE_99,
        ),
        ProblemSpec(
            pid="synth_100",
            title="Historic Peak Ledger",
            rating=2900,
            tags=["data structures", "segment tree", "segtree beats"],
            notes="Jiry (add, chmax) tropical semigroup historic maximum segment tree.",
            story_and_spec="""A reservoir control system monitors N gauges A_1, ..., A_N and also maintains for each gauge i its historic peak B_i, initialized to B_i = A_i.

Process Q operations of four types:
- 1 L R X : for each i in [L, R], set A_i <- A_i + X, then B_i <- max(B_i, A_i).
- 2 L R V : for each i in [L, R], set A_i <- max(A_i, V), then B_i <- max(B_i, A_i).
- 3 L R V : for each i in [L, R], set A_i <- V, then B_i <- max(B_i, A_i).
- 4 L R   : print max_{L <= i <= R} A_i and max_{L <= i <= R} B_i.

Input
The first line contains N and Q.
The second line contains initial values A_1 ... A_N.
Each of the next Q lines describes an operation.

Output
For each type-4 operation print two space-separated integers: the current range maximum and the historic peak range maximum.

Constraints
1 <= N, Q <= 150000
-10^8 <= A_i, X, V <= 10^8
1 <= L <= R <= N""",
            samples=[
                "5 7\n1 4 2 -3 5\n1 1 4 3\n4 1 5\n3 2 5 -10\n4 2 5\n2 1 5 6\n1 3 5 4\n4 1 5\n",
                "3 4\n-5 -2 -8\n1 1 3 -4\n4 1 3\n2 2 3 0\n4 1 3\n",
            ],
            small_inputs=[gen_100(2800, 2800, s, wide=True) for s in range(1, 5)],
            large_inputs=[
                gen_100(150000, 150000, 501, wide=True),
                gen_100(150000, 150000, 502, wide=False),
                gen_100(140000, 150000, 503, wide=True),
                gen_100(150000, 145000, 504, wide=True),
            ],
            fast_cpp=FAST_100,
            brute_cpp=BRUTE_100,
        ),
    ]


if __name__ == "__main__":
    build_and_verify_batch(make_specs())
