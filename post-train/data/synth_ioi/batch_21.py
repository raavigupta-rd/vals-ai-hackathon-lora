"""Batch 21: synth_101 .. synth_105 (Grandmaster Wave V: CF 2700-2900)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from builder import ProblemSpec, build_and_verify_batch


# ---------------------------------------------------------------------------
# synth_101 — Equal-Pair K-Partition (Divide & Conquer DP + Sliding Window)
# ---------------------------------------------------------------------------
FAST_101 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    if (!(cin >> n >> k)) return 0;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<int> cnt(n + 1, 0);
    int curL = 1, curR = 0;
    long long cur_cost = 0;

    auto get_cost = [&](int L, int R) -> long long {
        while (curL > L) {
            --curL;
            cur_cost += cnt[a[curL]]++;
        }
        while (curR < R) {
            ++curR;
            cur_cost += cnt[a[curR]]++;
        }
        while (curL < L) {
            cur_cost -= --cnt[a[curL]];
            ++curL;
        }
        while (curR > R) {
            cur_cost -= --cnt[a[curR]];
            --curR;
        }
        return cur_cost;
    };

    vector<long long> dp_prev(n + 1, INF), dp_cur(n + 1, INF);
    dp_prev[0] = 0;
    for (int i = 1; i <= n; i++) dp_prev[i] = get_cost(1, i);

    auto solve = [&](auto& self, int l, int r, int optL, int optR) -> void {
        if (l > r) return;
        int mid = (l + r) >> 1;
        long long best_val = INF;
        int best_j = optL;
        int hi = min(mid - 1, optR);
        for (int j = optL; j <= hi; j++) {
            long long cand = dp_prev[j] + get_cost(j + 1, mid);
            if (cand < best_val) {
                best_val = cand;
                best_j = j;
            }
        }
        dp_cur[mid] = best_val;
        self(self, l, mid - 1, optL, best_j);
        self(self, mid + 1, r, best_j, optR);
    };

    for (int step = 2; step <= k; step++) {
        fill(dp_cur.begin(), dp_cur.end(), INF);
        solve(solve, step, n, step - 1, n - 1);
        dp_prev.swap(dp_cur);
    }
    cout << dp_prev[n] << '\n';
    return 0;
}
"""

BRUTE_101 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];

    vector<long long> dp_prev(n + 1, INF), dp_cur(n + 1, INF);
    vector<int> cnt(n + 1, 0);
    dp_prev[0] = 0;
    for (int step = 1; step <= k; step++) {
        fill(dp_cur.begin(), dp_cur.end(), INF);
        for (int i = step; i <= n; i++) {
            long long pairs = 0;
            for (int j = i; j >= step; j--) {
                pairs += cnt[a[j]]++;
                if (dp_prev[j - 1] + pairs < dp_cur[i]) {
                    dp_cur[i] = dp_prev[j - 1] + pairs;
                }
            }
            for (int j = step; j <= i; j++) cnt[a[j]] = 0;
        }
        dp_prev.swap(dp_cur);
    }
    cout << dp_prev[n] << '\n';
    return 0;
}
"""


def gen_101(n: int, k: int, alpha: int, seed: int) -> str:
    r = random.Random(seed)
    a = [r.randint(1, min(n, alpha)) for _ in range(n)]
    return f"{n} {k}\n" + " ".join(map(str, a)) + "\n"


# ---------------------------------------------------------------------------
# synth_102 — Disjoint Subset Convolution Queries (Ranked SOS Zeta/Mobius)
# ---------------------------------------------------------------------------
FAST_102 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int b, q;
    if (!(cin >> b >> q)) return 0;
    int N = 1 << b;
    vector<vector<int>> fa(b + 1, vector<int>(N, 0));
    vector<vector<int>> fb(b + 1, vector<int>(N, 0));
    for (int m = 0; m < N; m++) {
        int x;
        cin >> x;
        fa[__builtin_popcount((unsigned)m)][m] = x % MOD;
    }
    for (int m = 0; m < N; m++) {
        int x;
        cin >> x;
        fb[__builtin_popcount((unsigned)m)][m] = x % MOD;
    }

    for (int r = 0; r <= b; r++) {
        auto& A = fa[r];
        auto& B = fb[r];
        for (int i = 0; i < b; i++) {
            int bit = 1 << i;
            for (int m = 0; m < N; m++) {
                if (m & bit) {
                    int u = A[m] + A[m ^ bit];
                    A[m] = (u >= MOD ? u - MOD : u);
                    int v = B[m] + B[m ^ bit];
                    B[m] = (v >= MOD ? v - MOD : v);
                }
            }
        }
    }

    vector<vector<int>> fc(b + 1, vector<int>(N, 0));
    for (int r = 0; r <= b; r++) {
        auto& C = fc[r];
        for (int j = 0; j <= r; j++) {
            const auto& A = fa[j];
            const auto& B = fb[r - j];
            for (int m = 0; m < N; m++) {
                C[m] = (int)((C[m] + 1LL * A[m] * B[m]) % MOD);
            }
        }
        for (int i = 0; i < b; i++) {
            int bit = 1 << i;
            for (int m = 0; m < N; m++) {
                if (m & bit) {
                    int u = C[m] - C[m ^ bit];
                    C[m] = (u < 0 ? u + MOD : u);
                }
            }
        }
    }

    while (q--) {
        int s;
        cin >> s;
        cout << fc[__builtin_popcount((unsigned)s)][s] << '\n';
    }
    return 0;
}
"""

BRUTE_102 = r"""
#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int b, q;
    cin >> b >> q;
    int N = 1 << b;
    vector<int> f(N), g(N);
    for (int i = 0; i < N; i++) {
        cin >> f[i];
        f[i] %= MOD;
    }
    for (int i = 0; i < N; i++) {
        cin >> g[i];
        g[i] %= MOD;
    }
    while (q--) {
        int s;
        cin >> s;
        long long ans = 0;
        for (int t = s; ; t = (t - 1) & s) {
            ans = (ans + 1LL * f[t] * g[s ^ t]) % MOD;
            if (t == 0) break;
        }
        cout << ans << '\n';
    }
    return 0;
}
"""


def gen_102(b: int, q: int, seed: int, heavy_masks: bool = False) -> str:
    r = random.Random(seed)
    mod = 998244353
    n = 1 << b
    f = [r.randint(0, mod - 1) for _ in range(n)]
    g = [r.randint(0, mod - 1) for _ in range(n)]
    qs = []
    full = n - 1
    for _ in range(q):
        if heavy_masks and r.random() < 0.85:
            # Flip at most 2 bits off full mask so popcount >= b-2
            m = full
            for _ in range(r.randint(0, 2)):
                m ^= 1 << r.randrange(b)
            qs.append(m)
        else:
            qs.append(r.randrange(n))
    return (
        f"{b} {q}\n"
        + " ".join(map(str, f))
        + "\n"
        + " ".join(map(str, g))
        + "\n"
        + "\n".join(map(str, qs))
        + "\n"
    )


# ---------------------------------------------------------------------------
# synth_103 — Rotational Mosaic Bracelets (Burnside + Euler Phi + Matrix Exp)
# ---------------------------------------------------------------------------
FAST_103 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

using Mat = array<array<int, 20>, 20>;

Mat mat_mul(const Mat& A, const Mat& B, int k) {
    Mat C{};
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            __int128 acc = 0;
            for (int t = 0; t < k; t++) {
                acc += (__int128)A[i][t] * B[t][j];
            }
            C[i][j] = (int)(acc % MOD);
        }
    }
    return C;
}

long long mod_pow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

long long euler_phi(long long x) {
    long long res = x;
    for (long long p = 2; p * p <= x; p++) {
        if (x % p == 0) {
            while (x % p == 0) x /= p;
            res -= res / p;
        }
    }
    if (x > 1) res -= res / x;
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    long long n;
    int k;
    if (!(cin >> n >> k)) return 0;
    Mat M{};
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            cin >> M[i][j];
            M[i][j] %= MOD;
        }
    }

    array<Mat, 31> pw;
    pw[0] = M;
    for (int b = 1; b <= 30; b++) pw[b] = mat_mul(pw[b - 1], pw[b - 1], k);

    auto trace_pow = [&](long long d) -> long long {
        Mat R{};
        for (int i = 0; i < k; i++) R[i][i] = 1;
        for (int b = 0; b <= 30; b++) {
            if ((d >> b) & 1LL) R = mat_mul(R, pw[b], k);
        }
        long long tr = 0;
        for (int i = 0; i < k; i++) tr = (tr + R[i][i]) % MOD;
        return tr;
    };

    vector<long long> divs;
    for (long long d = 1; d * d <= n; d++) {
        if (n % d == 0) {
            divs.push_back(d);
            if (d * d != n) divs.push_back(n / d);
        }
    }

    long long total = 0;
    for (long long d : divs) {
        long long phi = euler_phi(n / d) % MOD;
        long long tr = trace_pow(d);
        total = (total + phi * tr) % MOD;
    }
    long long ans = total * mod_pow(n % MOD, MOD - 2) % MOD;
    cout << ans << '\n';
    return 0;
}
"""

BRUTE_103 = r"""
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

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
    long long n;
    int k;
    cin >> n >> k;
    vector<vector<int>> M(k, vector<int>(k)), cur(k, vector<int>(k, 0)), nxt(k, vector<int>(k, 0));
    for (int i = 0; i < k; i++) {
        cur[i][i] = 1;
        for (int j = 0; j < k; j++) cin >> M[i][j];
    }

    // Step-by-step matrix powers O(N * K^3) time, O(K^2) memory
    long long total = 0;
    for (long long step = 1; step <= n; step++) {
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                long long s = 0;
                for (int t = 0; t < k; t++) s = (s + 1LL * cur[i][t] * M[t][j]) % MOD;
                nxt[i][j] = (int)s;
            }
        }
        cur.swap(nxt);
        if (n % step == 0) {
            long long tr = 0;
            for (int i = 0; i < k; i++) tr = (tr + cur[i][i]) % MOD;
            long long x = n / step, phi = x;
            for (long long p = 2; p * p <= x; p++) {
                if (x % p == 0) {
                    while (x % p == 0) x /= p;
                    phi -= phi / p;
                }
            }
            if (x > 1) phi -= phi / x;
            total = (total + (phi % MOD) * tr) % MOD;
        }
    }
    cout << total * mod_pow(n % MOD, MOD - 2) % MOD << '\n';
    return 0;
}
"""


def gen_103(n: int, k: int, seed: int) -> str:
    r = random.Random(seed)
    lines = [f"{n} {k}"]
    for _ in range(k):
        row = [1 if r.random() < 0.65 else 0 for _ in range(k)]
        lines.append(" ".join(map(str, row)))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_104 — Dynamic Conduit XOR Walks (Online Per-Component Basis + DSU)
# ---------------------------------------------------------------------------
FAST_104 = r"""
#include <bits/stdc++.h>
using namespace std;

struct Basis60 {
    unsigned long long b[60]{};

    void insert(unsigned long long x) {
        for (int i = 59; i >= 0; i--) {
            if (!((x >> i) & 1ULL)) continue;
            if (!b[i]) {
                b[i] = x;
                return;
            }
            x ^= b[i];
        }
    }

    unsigned long long max_xor(unsigned long long x) const {
        for (int i = 59; i >= 0; i--) {
            if ((x ^ b[i]) > x) x ^= b[i];
        }
        return x;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> par(n + 1), sz(n + 1, 1);
    vector<unsigned long long> pot(n + 1, 0);
    vector<Basis60> basis(n + 1);
    iota(par.begin(), par.end(), 0);

    auto find_root = [&](int u) -> pair<int, unsigned long long> {
        unsigned long long xr = 0;
        while (par[u] != u) {
            xr ^= pot[u];
            u = par[u];
        }
        return {u, xr};
    };

    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u, v;
            unsigned long long w;
            cin >> u >> v >> w;
            auto [ru, xu] = find_root(u);
            auto [rv, xv] = find_root(v);
            if (ru == rv) {
                basis[ru].insert(xu ^ xv ^ w);
            } else {
                if (sz[ru] < sz[rv]) {
                    swap(ru, rv);
                    swap(xu, xv);
                }
                par[rv] = ru;
                pot[rv] = xu ^ xv ^ w;
                sz[ru] += sz[rv];
                for (int i = 0; i < 60; i++) {
                    if (basis[rv].b[i]) basis[ru].insert(basis[rv].b[i]);
                }
            }
        } else {
            int u, v;
            cin >> u >> v;
            auto [ru, xu] = find_root(u);
            auto [rv, xv] = find_root(v);
            if (ru != rv) {
                cout << -1 << '\n';
            } else {
                cout << basis[ru].max_xor(xu ^ xv) << '\n';
            }
        }
    }
    return 0;
}
"""

BRUTE_104 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<pair<int, unsigned long long>>> adj(n + 1);
    vector<int> vis(n + 1, 0), que(n + 1);
    vector<unsigned long long> dist(n + 1, 0);
    int timer = 0;

    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u, v;
            unsigned long long w;
            cin >> u >> v >> w;
            adj[u].push_back({v, w});
            adj[v].push_back({u, w});
        } else {
            int s, t;
            cin >> s >> t;
            ++timer;
            int head = 0, tail = 0;
            vis[s] = timer;
            dist[s] = 0;
            que[tail++] = s;
            unsigned long long b[60]{};
            auto ins = [&](unsigned long long x) {
                for (int i = 59; i >= 0; i--) {
                    if (!((x >> i) & 1ULL)) continue;
                    if (!b[i]) {
                        b[i] = x;
                        return;
                    }
                    x ^= b[i];
                }
            };
            while (head < tail) {
                int u = que[head++];
                for (auto [v, w] : adj[u]) {
                    if (vis[v] != timer) {
                        vis[v] = timer;
                        dist[v] = dist[u] ^ w;
                        que[tail++] = v;
                    } else {
                        ins(dist[u] ^ dist[v] ^ w);
                    }
                }
            }
            if (vis[t] != timer) {
                cout << -1 << '\n';
            } else {
                unsigned long long ans = dist[t];
                for (int i = 59; i >= 0; i--) {
                    if ((ans ^ b[i]) > ans) ans ^= b[i];
                }
                cout << ans << '\n';
            }
        }
    }
    return 0;
}
"""


def gen_104(n: int, q: int, seed: int) -> str:
    r = random.Random(seed)
    lines = [f"{n} {q}"]
    # Maintain 3-5 rich clusters early so naive single-global-basis solutions fail,
    # then gradually connect them while inserting many cycles.
    clusters = 4
    for step in range(q):
        if step < q // 4 or r.random() < 0.6:
            if step < (2 * q) // 3 and r.random() < 0.85:
                cid = r.randrange(clusters)
                lo = (cid * n) // clusters + 1
                hi = max(lo, ((cid + 1) * n) // clusters)
                u = r.randint(lo, hi)
                v = r.randint(lo, hi)
            else:
                u = r.randint(1, n)
                v = r.randint(1, n)
            w = r.randrange(1 << 60)
            lines.append(f"1 {u} {v} {w}")
        else:
            if r.random() < 0.75:
                cid = r.randrange(clusters)
                lo = (cid * n) // clusters + 1
                hi = max(lo, ((cid + 1) * n) // clusters)
                u = r.randint(lo, hi)
                v = r.randint(lo, hi)
            else:
                u = r.randint(1, n)
                v = r.randint(1, n)
            lines.append(f"2 {u} {v}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# synth_105 — All-Roots Tree Isomorphism Classes (Rerooting 128-bit Tree Hash)
# ---------------------------------------------------------------------------
FAST_105 = r"""
#include <bits/stdc++.h>
using namespace std;

using u64 = unsigned long long;

static inline u64 splitmix1(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline u64 splitmix2(u64 x) {
    x += 0x243f6a8885a308d3ULL;
    x = (x ^ (x >> 29)) * 0x13198a2e03707344ULL;
    x = (x ^ (x >> 26)) * 0xa4093822299f31d0ULL;
    return x ^ (x >> 32);
}

struct Hash128 {
    u64 h1, h2;
    bool operator<(const Hash128& o) const {
        return h1 != o.h1 ? h1 < o.h1 : h2 < o.h2;
    }
    bool operator==(const Hash128& o) const {
        return h1 == o.h1 && h2 == o.h2;
    }
};

static inline Hash128 wrap_child(const Hash128& c, int sz) {
    u64 a = splitmix1(c.h1 ^ (c.h2 >> 13) ^ ((u64)sz * 0x517cc1b727220a95ULL));
    u64 b = splitmix2(c.h2 ^ (c.h1 >> 11) ^ ((u64)sz * 0x9e3779b97f4a7c15ULL));
    return {a, b};
}

static inline Hash128 finalize_sum(u64 s1, u64 s2, int sz) {
    u64 a = splitmix1(s1 + 0x6a09e667f3bcc909ULL + (u64)sz);
    u64 b = splitmix2(s2 + 0xbb67ae8584caa73bULL + (u64)sz * 3ULL);
    return {a, b};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), ord, sz(n + 1, 1);
    ord.reserve(n);
    ord.push_back(1);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) {
            if (v != par[u]) {
                par[v] = u;
                ord.push_back(v);
            }
        }
    }

    vector<u64> sum1(n + 1, 0), sum2(n + 1, 0);
    vector<Hash128> down(n + 1), root_hash(n + 1);
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        u64 s1 = 0, s2 = 0;
        for (int v : adj[u]) {
            if (v == par[u]) continue;
            sz[u] += sz[v];
            Hash128 w = wrap_child(down[v], sz[v]);
            s1 += w.h1;
            s2 += w.h2;
        }
        sum1[u] = s1;
        sum2[u] = s2;
        down[u] = finalize_sum(s1, s2, sz[u]);
    }

    vector<Hash128> up_wrapped(n + 1, {0, 0});
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        u64 tot1 = sum1[u] + up_wrapped[u].h1;
        u64 tot2 = sum2[u] + up_wrapped[u].h2;
        root_hash[u] = finalize_sum(tot1, tot2, n);
        for (int v : adj[u]) {
            if (v == par[u]) continue;
            Hash128 wv = wrap_child(down[v], sz[v]);
            u64 rem1 = tot1 - wv.h1;
            u64 rem2 = tot2 - wv.h2;
            Hash128 up_sub = finalize_sum(rem1, rem2, n - sz[v]);
            up_wrapped[v] = wrap_child(up_sub, n - sz[v]);
        }
    }

    map<Hash128, int> id_map;
    vector<int> ans(n + 1);
    int next_id = 0;
    for (int u = 1; u <= n; u++) {
        auto [it, inserted] = id_map.emplace(root_hash[u], next_id + 1);
        if (inserted) ++next_id;
        ans[u] = it->second;
    }

    cout << next_id << '\n';
    for (int u = 1; u <= n; u++) {
        cout << ans[u] << (u == n ? '\n' : ' ');
    }
    return 0;
}
"""

BRUTE_105 = r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Exact deterministic AHU tuple-ID table shared across all roots
    map<vector<int>, int> ahu;
    auto get_ahu_id = [&](vector<int>& ch) -> int {
        sort(ch.begin(), ch.end());
        auto [it, ins] = ahu.emplace(ch, (int)ahu.size() + 1);
        return it->second;
    };

    vector<int> par(n + 1), ord(n), sub_id(n + 1), raw_root(n + 1);
    for (int r = 1; r <= n; r++) {
        par[r] = 0;
        int head = 0, tail = 0;
        ord[tail++] = r;
        while (head < tail) {
            int u = ord[head++];
            for (int v : adj[u]) {
                if (v != par[u]) {
                    par[v] = u;
                    ord[tail++] = v;
                }
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            int u = ord[i];
            vector<int> ch;
            for (int v : adj[u]) {
                if (v != par[u]) ch.push_back(sub_id[v]);
            }
            sub_id[u] = get_ahu_id(ch);
        }
        raw_root[r] = sub_id[r];
    }

    map<int, int> remap;
    int cnt = 0;
    vector<int> ans(n + 1);
    for (int u = 1; u <= n; u++) {
        auto [it, ins] = remap.emplace(raw_root[u], cnt + 1);
        if (ins) ++cnt;
        ans[u] = it->second;
    }
    cout << cnt << '\n';
    for (int u = 1; u <= n; u++) cout << ans[u] << (u == n ? '\n' : ' ');
    return 0;
}
"""


def gen_105(n: int, seed: int, symmetric: bool = False) -> str:
    r = random.Random(seed)
    edges = []
    if symmetric and n >= 15:
        # Build a tree with rich automorphisms: central cycle-free hub with identical gadgets
        hub_children = max(3, min(12, n // 20))
        # Build a random small gadget template of size g
        g = (n - 1) // hub_children
        tmpl = [(r.randint(0, i - 1), i) for i in range(1, g)]
        cur = 2
        for _ in range(hub_children):
            base = cur
            edges.append((1, base))
            for a, b in tmpl:
                edges.append((base + a, base + b))
            cur += g
        while cur <= n:
            edges.append((1, cur))
            cur += 1
    else:
        for i in range(2, n + 1):
            edges.append((r.randint(1, i - 1), i))
    perm = list(range(1, n + 1))
    r.shuffle(perm)
    lines = [str(n)]
    for u, v in edges:
        lines.append(f"{perm[u - 1]} {perm[v - 1]}")
    return "\n".join(lines) + "\n"


def make_specs() -> list[ProblemSpec]:
    return [
        ProblemSpec(
            pid="synth_101",
            title="Gondola Clans",
            rating=2700,
            tags=["dp", "divide and conquer"],
            notes="CF 868F: Divide-and-conquer DP optimization with two-pointer Mo-style frequency window.",
            story_and_spec="""N passengers wait in line for K gondolas, numbered 1 to N in queue order. Passenger i belongs to clan A_i.

You must partition the queue into K non-empty contiguous groups so each gondola carries a contiguous segment [L_t, R_t] of passengers. Inside a single gondola [L, R], every pair of passengers from the same clan (L <= i < j <= R with A_i = A_j) generates 1 unit of chatter.

Find the minimum possible total chatter summed across all K gondolas.

Input
The first line contains N and K.
The second line contains A_1 ... A_N.

Output
Print the minimum total chatter.

Constraints
1 <= N <= 70000
1 <= K <= min(N, 25)
1 <= A_i <= N""",
            samples=[
                "7 3\n1 1 3 3 3 2 1\n",
                "6 2\n1 2 1 2 1 2\n",
            ],
            small_inputs=[
                gen_101(1500, 12, 10, 1),
                gen_101(1700, 15, 50, 2),
                gen_101(1600, 1, 8, 3),
                gen_101(1400, 20, 4, 4),
            ],
            large_inputs=[
                gen_101(70000, 20, 30, 101),
                gen_101(70000, 25, 5, 102),
                gen_101(68000, 25, 500, 103),
                gen_101(70000, 18, 10000, 104),
            ],
            fast_cpp=FAST_101,
            brute_cpp=BRUTE_101,
        ),
        ProblemSpec(
            pid="synth_102",
            title="Disjoint Subset Resonance",
            rating=2800,
            tags=["bitmasks", "dp", "sos", "math"],
            notes="Disjoint subset convolution via ranked SOS Zeta and Mobius transforms in O(B^2 2^B).",
            story_and_spec="""An alchemist has B elemental catalysts numbered 0 to B - 1. Every subset of catalysts is represented by an integer bitmask S in [0, 2^B - 1].

You are given two arrays F[0 ... 2^B - 1] and G[0 ... 2^B - 1]. For each subset S, define the disjoint subset resonance:
H[S] = (sum_{T subseteq S} F[T] * G[S \\ T]) mod 998244353,
where S \\ T denotes the set difference (equivalently, T & (S ^ T) == 0 and T | (S ^ T) == S).

Given Q queried bitmasks S_1, ..., S_Q, print H[S_j] for each query.

Input
The first line contains B and Q.
The second line contains 2^B integers F[0] ... F[2^B - 1].
The third line contains 2^B integers G[0] ... G[2^B - 1].
Each of the next Q lines contains an integer S_j (0 <= S_j < 2^B).

Output
For each query print H[S_j] on its own line.

Constraints
1 <= B <= 19
1 <= Q <= 100000
0 <= F[i], G[i] < 998244353""",
            samples=[
                "2 4\n1 2 3 4\n5 6 7 8\n0\n1\n2\n3\n",
                "3 3\n1 1 1 1 1 1 1 1\n1 2 3 4 5 6 7 8\n7\n5\n0\n",
            ],
            small_inputs=[
                gen_102(11, 2048, 1, heavy_masks=True),
                gen_102(12, 2500, 2, heavy_masks=True),
                gen_102(10, 1024, 3, heavy_masks=False),
                gen_102(1, 10, 4, heavy_masks=True),
            ],
            large_inputs=[
                gen_102(19, 100000, 201, heavy_masks=True),
                gen_102(19, 100000, 202, heavy_masks=False),
                gen_102(18, 100000, 203, heavy_masks=True),
                gen_102(19, 90000, 204, heavy_masks=True),
            ],
            fast_cpp=FAST_102,
            brute_cpp=BRUTE_102,
        ),
        ProblemSpec(
            pid="synth_103",
            title="Rotational Mosaic Bracelets",
            rating=2700,
            tags=["math", "combinatorics", "matrices", "number theory"],
            notes="Burnside's Lemma over cyclic group C_N with Euler totient and binary matrix exponentiation tr(M^d).",
            story_and_spec="""A jeweler crafts circular bracelets consisting of N beads arranged in a ring, each chosen from K gem types numbered 1 to K. You are given a K x K compatibility matrix M where M_{u,v} in {0, 1} equals 1 if gem type u may be followed immediately clockwise by gem type v, and 0 otherwise (including bead N followed clockwise by bead 1).

Two valid bracelets are considered identical if one can be obtained from the other by a cyclic rotation (rotations by 0, 1, ..., N - 1 positions; reflections are NOT allowed).

Count the number of distinct valid bracelets modulo 998244353.

Input
The first line contains N and K.
Each of the next K lines contains K integers M_{u,1} ... M_{u,K} (each 0 or 1).

Output
Print the number of distinct bracelets modulo 998244353.

Constraints
1 <= N <= 900000000
1 <= K <= 20
M_{u,v} in {0, 1}""",
            samples=[
                "4 2\n1 1\n1 0\n",
                "6 3\n0 1 1\n1 0 1\n1 1 1\n",
            ],
            small_inputs=[
                gen_103(1200, 18, 1),
                gen_103(1440, 20, 2),
                gen_103(997, 15, 3),
                gen_103(1, 20, 4),
            ],
            large_inputs=[
                gen_103(720720000, 20, 301),
                gen_103(899999993, 20, 302),
                gen_103(735134400, 20, 303),
                gen_103(840000000, 19, 304),
            ],
            fast_cpp=FAST_103,
            brute_cpp=BRUTE_103,
        ),
        ProblemSpec(
            pid="synth_104",
            title="Dynamic Conduit XOR Walks",
            rating=2800,
            tags=["dsu", "linear algebra", "bitmasks", "graphs"],
            notes="Online DSU with XOR potential and merged per-component 60-bit cycle-space linear bases.",
            story_and_spec="""An energy grid consists of N stations numbered 1 to N and initially has 0 conduits.

Process Q events of two types:
- 1 u v w : install a bidirectional conduit between stations u and v with 60-bit phase weight w (parallel conduits and self-loops are allowed).
- 2 u v   : if no walk currently connects u and v, print -1. Otherwise print the maximum possible bitwise XOR of conduit weights along a walk starting at u and ending at v (a walk may repeat stations and conduits arbitrarily many times; each traversal of a conduit XORs its weight w once).

Input
The first line contains N and Q.
Each of the next Q lines describes an event in one of the two formats above.

Output
For each type-2 event print the answer on its own line.

Constraints
1 <= N <= 100000
1 <= Q <= 150000
0 <= w < 2^60""",
            samples=[
                "5 8\n1 1 2 3\n1 2 3 5\n1 3 1 10\n1 4 5 64\n2 1 3\n2 1 5\n1 3 4 1\n2 1 5\n",
                "3 5\n2 1 1\n1 1 1 7\n2 1 1\n1 1 2 4\n2 1 2\n",
            ],
            small_inputs=[gen_104(1800, 2200, s) for s in range(1, 5)],
            large_inputs=[
                gen_104(100000, 150000, 401),
                gen_104(100000, 150000, 402),
                gen_104(95000, 150000, 403),
                gen_104(100000, 140000, 404),
            ],
            fast_cpp=FAST_104,
            brute_cpp=BRUTE_104,
        ),
        ProblemSpec(
            pid="synth_105",
            title="Rooted Canopy Symmetries",
            rating=2700,
            tags=["trees", "hashing", "dp"],
            notes="All-roots rooted tree isomorphism via rerooting 128-bit SplitMix64 symmetric tree hashes.",
            story_and_spec="""A botanical researcher studies an unrooted tree of N junctions numbered 1 to N. For each junction r (1 <= r <= N), let T_r denote the rooted tree obtained by choosing r as the root.

Two junctions u and v belong to the same structural symmetry class if T_u and T_v are isomorphic as rooted unlabeled trees.

Determine the total number C of distinct structural symmetry classes, and assign each junction u an integer class label I_u in {1, ..., C} normalized by first appearance: junction 1 receives label 1, and as u increases from 1 to N, each junction whose rooted tree is not isomorphic to any T_j (j < u) receives the next unused positive integer.

Input
The first line contains N.
Each of the next N - 1 lines contains two integers u and v describing an edge.

Output
On the first line print C.
On the second line print I_1 I_2 ... I_N separated by spaces.

Constraints
1 <= N <= 180000""",
            samples=[
                "5\n1 2\n2 3\n3 4\n4 5\n",
                "6\n1 2\n1 3\n2 4\n2 5\n3 6\n",
            ],
            small_inputs=[
                gen_105(1100, 1, symmetric=True),
                gen_105(1200, 2, symmetric=False),
                gen_105(1000, 3, symmetric=True),
                "1\n",
            ],
            large_inputs=[
                gen_105(180000, 501, symmetric=True),
                gen_105(180000, 502, symmetric=False),
                gen_105(175000, 503, symmetric=True),
                gen_105(180000, 504, symmetric=False),
            ],
            fast_cpp=FAST_105,
            brute_cpp=BRUTE_105,
        ),
    ]


if __name__ == "__main__":
    build_and_verify_batch(make_specs())
