"""Batch 12 (synth_56 .. synth_60): Dynamic Programming II — LIS Length & Count, WQS Aliens Disjoint Windows, Bounded Subset Min-Count Knapsack, Composite-Modulus Rerooting, Divide-and-Conquer Min*Max Subarrays."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def _rand_tree(n: int, r: random.Random, kind: str = "random") -> list[tuple[int, int]]:
    edges = []
    if kind == "chain":
        perm = list(range(1, n + 1))
        r.shuffle(perm)
        for i in range(n - 1):
            edges.append((perm[i], perm[i + 1]))
    elif kind == "star_clusters":
        # Many degree-2 leaves so (down[c] + 1) = 2 shares factors with even M
        spine = max(2, n // 4)
        for i in range(2, spine + 1):
            edges.append((i - 1, i))
        for i in range(spine + 1, n + 1):
            edges.append((r.randint(1, spine), i))
    else:
        perm = list(range(1, n + 1))
        r.shuffle(perm)
        for i in range(1, n):
            p = perm[r.randint(0, i - 1)]
            edges.append((p, perm[i]))
    return edges


def make_batch_12() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_56: FORMATION RANKS (1900) - LIS Length and Count via (len, ways) BIT
    # ------------------------------------------------------------------
    def gen_56(n: int, vmax: int, seed: int, layered: bool = False) -> str:
        r = random.Random(seed)
        if layered:
            # Blocks of decreasing values with jitter -> huge number of maximum-length LIS
            b = max(2, int(n ** 0.5))
            a = []
            for i in range(n):
                grp = i // b
                a.append(grp * b + (b - (i % b)) + r.randint(0, 2))
        else:
            a = [r.randint(1, vmax) for _ in range(n)]
        return f"{n}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_56",
        title="Formation Ranks",
        rating=1900,
        tags=["dp", "data structures", "fenwick tree", "coordinate compression"],
        notes="Coordinate-compress A and maintain a Fenwick tree over pairs (max_len, ways_mod) to compute both LIS length and count in O(N log N).",
        story_and_spec="""N soldiers stand in a line with skill ratings A_1, A_2, ..., A_N. You want to select a subsequence of indices 1 <= i_1 < i_2 < ... < i_L <= N such that their ratings are strictly increasing:
A_{i_1} < A_{i_2} < ... < A_{i_L}.

Find BOTH:
1. L_max, the maximum possible length of such a strictly increasing subsequence.
2. C, the number of distinct index subsequences (i_1, ..., i_{L_max}) that achieve length L_max, modulo 10^9 + 7.

Input
The first line contains N.
The second line contains A_1 ... A_N.

Output
Print two space-separated integers: L_max and C (modulo 10^9 + 7).

Constraints
1 <= N <= 200000
1 <= A_i <= 10^9""",
        samples=[
            "6\n1 3 2 4 3 5\n",
            "4\n7 7 7 7\n",
        ],
        small_inputs=[
            "1\n42\n",
            gen_56(400, 50, 5601),
            gen_56(2200, 10**9, 5602, layered=True),
        ],
        large_inputs=[
            gen_56(200000, 10**9, 5611),
            gen_56(200000, 5000, 5612),
            gen_56(200000, 10**9, 5613, layered=True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

struct Val {
    int len = 0;
    int cnt = 0;
};

static inline Val merge_val(Val a, Val b) {
    if (a.len != b.len) return a.len > b.len ? a : b;
    int s = a.cnt + b.cnt;
    if (s >= MOD) s -= MOD;
    return {a.len, s};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    vector<int> vals = a;
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();

    vector<Val> bit(m + 1);
    auto query = [&](int idx) {
        Val res{0, 0};
        for (; idx > 0; idx -= idx & -idx) res = merge_val(res, bit[idx]);
        return res;
    };
    auto update = [&](int idx, Val v) {
        for (; idx <= m; idx += idx & -idx) bit[idx] = merge_val(bit[idx], v);
    };

    for (int x : a) {
        int rk = (int)(lower_bound(vals.begin(), vals.end(), x) - vals.begin()) + 1;
        Val best = query(rk - 1);
        Val cur = (best.len == 0) ? Val{1, 1} : Val{best.len + 1, best.cnt};
        update(rk, cur);
    }
    Val ans = query(m);
    cout << ans.len << ' ' << ans.cnt << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n), len(n, 1), cnt(n, 1);
    for (int i = 0; i < n; i++) cin >> a[i];
    int best_len = 0, best_cnt = 0;
    for (int i = 0; i < n; i++) {
        int l = 1, c = 1;
        for (int j = 0; j < i; j++) {
            if (a[j] < a[i]) {
                if (len[j] + 1 > l) {
                    l = len[j] + 1;
                    c = cnt[j];
                } else if (len[j] + 1 == l) {
                    c += cnt[j];
                    if (c >= MOD) c -= MOD;
                }
            }
        }
        len[i] = l;
        cnt[i] = c;
        if (l > best_len) {
            best_len = l;
            best_cnt = c;
        } else if (l == best_len) {
            best_cnt += c;
            if (best_cnt >= MOD) best_cnt -= MOD;
        }
    }
    cout << best_len << ' ' << best_cnt << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_57: SIGNAL WINDOWS (2300) - WQS (Aliens) Binary Search
    # ------------------------------------------------------------------
    def gen_57(n: int, k: int, length: int, vmin: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(vmin, vmax) for _ in range(n)]
        return f"{n} {k} {length}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_57",
        title="Signal Windows",
        rating=2300,
        tags=["dp", "wqs binary search", "aliens trick", "convexity"],
        notes="Value as a function of window count k is concave over integers; WQS binary search on non-negative per-window penalty lambda reduces O(NK) DP to O(N log(sum|A|)).",
        story_and_spec="""A radio telescope records N consecutive signal measurements A_1, A_2, ..., A_N (which may be positive, zero, or negative).

You may choose AT MOST K pairwise non-overlapping contiguous windows, where EVERY chosen window has exact length L. That is, you choose m (0 <= m <= K) starting indices s_1, s_2, ..., s_m such that:
- 1 <= s_1,
- s_j + L <= s_{j+1} for all 1 <= j < m, and
- s_m + L - 1 <= N.

Find the maximum possible total sum of all measurements inside your chosen windows (choosing m = 0 windows gives a sum of 0).

Input
The first line contains N, K, and L.
The second line contains A_1 ... A_N.

Output
Print the maximum achievable total sum.

Constraints
1 <= L <= N <= 200000
1 <= K <= 60000
-10^6 <= A_i <= 10^6""",
        samples=[
            "8 2 3\n2 -1 4 3 -2 5 1 -3\n",
            "5 3 2\n-4 -2 -5 -1 -3\n",
        ],
        small_inputs=[
            "6 2 2\n5 5 5 5 5 5\n",
            gen_57(400, 80, 3, -1000, 1500, 5701),
            gen_57(1500, 400, 2, -10**6, 10**6, 5702),
            gen_57(1200, 300, 7, -500, 2000, 5703),
        ],
        large_inputs=[
            gen_57(180000, 50000, 2, -10**6, 10**6, 5711),
            gen_57(200000, 40000, 3, -10**5, 10**6, 5712),
            gen_57(200000, 25000, 5, -10**6, 10**6, 5713),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, L;
    cin >> n >> k >> L;
    vector<long long> pref(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        pref[i] = pref[i - 1] + x;
    }
    vector<long long> dp_val(n + 1, 0);
    vector<int> dp_cnt(n + 1, 0);

    // Solve penalized DP: maximize (sum - lam * count), breaking ties toward MINIMUM count.
    auto solve_pen = [&](long long lam) -> pair<long long, int> {
        dp_val[0] = 0;
        dp_cnt[0] = 0;
        for (int i = 1; i <= n; i++) {
            dp_val[i] = dp_val[i - 1];
            dp_cnt[i] = dp_cnt[i - 1];
            if (i >= L) {
                long long cand_v = dp_val[i - L] + (pref[i] - pref[i - L]) - lam;
                int cand_c = dp_cnt[i - L] + 1;
                if (cand_v > dp_val[i] || (cand_v == dp_val[i] && cand_c < dp_cnt[i])) {
                    dp_val[i] = cand_v;
                    dp_cnt[i] = cand_c;
                }
            }
        }
        return {dp_val[n], dp_cnt[n]};
    };

    auto [v0, c0] = solve_pen(0);
    if (c0 <= k) {
        cout << v0 << '\n';
        return 0;
    }

    long long mx_w = 0;
    for (int i = L; i <= n; i++) mx_w = max(mx_w, pref[i] - pref[i - L]);
    long long lo = 1, hi = mx_w + 1;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (solve_pen(mid).second <= k) hi = mid;
        else lo = mid + 1;
    }
    auto [v_opt, _] = solve_pen(lo);
    cout << v_opt + lo * 1LL * k << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, L;
    cin >> n >> k >> L;
    vector<long long> pref(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        pref[i] = pref[i - 1] + x;
    }
    k = min(k, n / L);
    vector<long long> prv(n + 1, 0), cur(n + 1, 0);
    for (int j = 1; j <= k; j++) {
        fill(cur.begin(), cur.begin() + L, 0);
        for (int i = L; i <= n; i++) {
            cur[i] = max(cur[i - 1], prv[i - L] + (pref[i] - pref[i - L]));
        }
        swap(prv, cur);
    }
    cout << prv[n] << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_58: BALANCED STONES (2000) - Bounded Knapsack via Binary Splitting
    # ------------------------------------------------------------------
    def gen_58(kind: str, seed: int) -> str:
        r = random.Random(seed)
        if kind == "adversarial_large":
            # 1 stone of 50000 + 190000 stones of 1 -> sum = 240000, naive loop does ~2.8e10 int ops
            w = [50000] + [1] * 190000
        elif kind == "sqrt_large":
            w = []
            tot = 0
            while tot < 235000 and len(w) < 180000:
                v = r.randint(1, 40) if r.random() < 0.9 else r.randint(41, 2000)
                if tot + v <= 240000:
                    w.append(v)
                    tot += v
                else:
                    break
            r.shuffle(w)
        elif kind == "many_twos_large":
            w = [30000, 20000] + [1] * 100000 + [2] * 45000
            r.shuffle(w)
            w = [40000] + [x for x in w if sum(w) >= 0][:170000]
            # Keep sum <= 240000
            out = [40000]
            s = 40000
            for x in ([1] * 110000 + [2] * 40000):
                if s + x <= 240000:
                    out.append(x)
                    s += x
            w = out
        elif kind == "small_rand":
            w = [r.randint(1, 30) for _ in range(80)]
        else:
            w = [500] + [r.randint(1, 5) for _ in range(900)]
        return f"{len(w)}\n" + " ".join(map(str, w)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_58",
        title="Balanced Stones",
        rating=2000,
        tags=["dp", "knapsack", "sqrt decomposition", "binary splitting"],
        notes="Since sum(W_i) <= 240,000, grouping identical weights and binary-splitting multiplicities yields <= ~1,200 bundles, reducing O(N * S) min-stone knapsack to O(S * sqrt(S)).",
        story_and_spec="""An alchemist has N stones with positive integer weights W_1, W_2, ..., W_N, whose total weight S = W_1 + W_2 + ... + W_N is at most 240000.

For each target weight x from 1 to S:
- Let f(x) be the MINIMUM number of stones needed to form a subset of total weight exactly x, or 0 if no subset of stones sums to x.

Compute two values:
1. K, the number of target weights x in [1, S] that CAN be formed.
2. T = f(1) + f(2) + ... + f(S), the sum of minimum stone counts across all achievable positive target weights.

Input
The first line contains N.
The second line contains W_1 ... W_N.

Output
Print two space-separated integers: K and T.

Constraints
1 <= N <= 200000
1 <= W_i <= 240000
W_1 + W_2 + ... + W_N <= 240000""",
        samples=[
            "5\n1 1 2 2 5\n",
        ],
        small_inputs=[
            "1\n7\n",
            gen_58("small_rand", 5801),
            gen_58("small_dense", 5802),
        ],
        large_inputs=[
            gen_58("adversarial_large", 5811),
            gen_58("sqrt_large", 5812),
            gen_58("many_twos_large", 5813),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    int S = 0;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        S += a[i];
    }
    vector<int> freq(S + 1, 0);
    for (int w : a) freq[w]++;

    vector<pair<int, int>> bundles;
    for (int w = 1; w <= S; w++) {
        int c = freq[w];
        for (int k = 1; c > 0; k <<= 1) {
            int take = min(k, c);
            bundles.push_back({take * w, take});
            c -= take;
        }
    }

    const int INF = 1e9;
    vector<int> dp(S + 1, INF);
    dp[0] = 0;
    int cur = 0;
    for (auto [bw, bc] : bundles) {
        cur += bw;
        for (int s = cur; s >= bw; s--) {
            dp[s] = min(dp[s], dp[s - bw] + bc);
        }
    }

    long long cnt_ok = 0, sum_f = 0;
    for (int s = 1; s <= S; s++) {
        if (dp[s] < INF) {
            cnt_ok++;
            sum_f += dp[s];
        }
    }
    cout << cnt_ok << ' ' << sum_f << '\n';
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
    int S = 0;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        S += a[i];
    }
    const int INF = 1e9;
    vector<int> dp(S + 1, INF);
    dp[0] = 0;
    int cur = 0;
    for (int w : a) {
        cur += w;
        for (int s = cur; s >= w; s--) {
            int cand = dp[s - w] + 1;
            if (cand < dp[s]) dp[s] = cand;
        }
    }
    long long cnt_ok = 0, sum_f = 0;
    for (int s = 1; s <= S; s++) {
        if (dp[s] < INF) {
            cnt_ok++;
            sum_f += dp[s];
        }
    }
    cout << cnt_ok << ' ' << sum_f << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_59: SUBTREE ALLIANCES (2000) - Rerooting DP Modulo Arbitrary M
    # ------------------------------------------------------------------
    def gen_59(n: int, mod: int, seed: int, kind: str = "random") -> str:
        r = random.Random(seed)
        es = _rand_tree(n, r, kind)
        return f"{n} {mod}\n" + "\n".join(f"{u} {v}" for u, v in es) + "\n"

    specs.append(ProblemSpec(
        pid="synth_59",
        title="Subtree Alliances",
        rating=2000,
        tags=["dp", "trees", "rerooting", "combinatorics"],
        notes="Rerooting tree DP modulo arbitrary composite M (no modular inverse): use prefix and suffix products over children's (down[c] + 1) factors.",
        story_and_spec="""A confederation consists of N cities numbered 1 to N connected by N - 1 bidirectional roads forming a tree. An alliance is a non-empty subset of cities that induces a connected subgraph of the tree.

Given an integer M (which is NOT necessarily prime), compute for EVERY city v = 1, 2, ..., N the number of valid alliances that include city v, modulo M.

Input
The first line contains N and M.
Each of the next N - 1 lines contains two cities U and V connected by a road.

Output
Print N lines: line v should contain the number of connected city subsets containing city v, modulo M.

Constraints
1 <= N <= 200000
2 <= M <= 10^9""",
        samples=[
            "4 100\n1 2\n1 3\n1 4\n",
            "4 6\n1 2\n2 3\n3 4\n",
        ],
        small_inputs=[
            "1 7\n",
            gen_59(300, 12, 5901, "star_clusters"),
            gen_59(1400, 1000000000, 5902, "chain"),
        ],
        large_inputs=[
            gen_59(200000, 1000000000, 5911, "chain"),
            gen_59(200000, 720720, 5912, "star_clusters"),
            gen_59(200000, 999999990, 5913, "random"),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long mod;
    cin >> n >> mod;
    vector<vector<int>> adj(n + 1), ch(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), ord;
    ord.reserve(n);
    ord.push_back(1);
    par[1] = -1;
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ch[u].push_back(v);
            ord.push_back(v);
        }
    }

    vector<long long> down(n + 1, 1), up_fac(n + 1, 1);
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        long long prod = 1;
        for (int c : ch[u]) prod = (prod * (down[c] + 1)) % mod;
        down[u] = prod;
    }

    for (int u : ord) {
        int deg = (int)ch[u].size();
        vector<long long> pref(deg + 1, 1), suff(deg + 1, 1);
        for (int i = 0; i < deg; i++) pref[i + 1] = (pref[i] * (down[ch[u][i]] + 1)) % mod;
        for (int i = deg - 1; i >= 0; i--) suff[i] = (suff[i + 1] * (down[ch[u][i]] + 1)) % mod;
        for (int i = 0; i < deg; i++) {
            int c = ch[u][i];
            long long u_without_c = up_fac[u] * pref[i] % mod * suff[i + 1] % mod;
            up_fac[c] = (u_without_c + 1) % mod;
        }
    }

    for (int u = 1; u <= n; u++) {
        cout << (down[u] * up_fac[u]) % mod << '\n';
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
    long long mod;
    cin >> n >> mod;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> par(n + 1), ord;
    vector<long long> dp(n + 1);
    ord.reserve(n);
    for (int root = 1; root <= n; root++) {
        ord.clear();
        fill(par.begin(), par.end(), 0);
        ord.push_back(root);
        par[root] = -1;
        for (int i = 0; i < n; i++) {
            int u = ord[i];
            for (int v : adj[u]) if (v != par[u]) {
                par[v] = u;
                ord.push_back(v);
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            int u = ord[i];
            long long prod = 1;
            for (int v : adj[u]) if (v != par[u]) {
                prod = (prod * (dp[v] + 1)) % mod;
            }
            dp[u] = prod;
        }
        cout << dp[root] % mod << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_60: EXTREMAL SUBARRAYS (2200) - Divide & Conquer Two-Pointer Sum of Min*Max
    # ------------------------------------------------------------------
    def gen_60(n: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        return f"{n}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_60",
        title="Extremal Subarrays",
        rating=2200,
        tags=["divide and conquer", "two pointers", "combinatorics"],
        notes="Divide & conquer on [lo, hi] with two monotone pointers p, q on the right half splitting min and max dominance into 3 prefix-summed regions in O(N log N).",
        story_and_spec="""Given an integer array A_1, A_2, ..., A_N, define for every contiguous subarray [L, R] (1 <= L <= R <= N):
- mx(L, R) = max(A_L, A_{L+1}, ..., A_R),
- mn(L, R) = min(A_L, A_{L+1}, ..., A_R).

Compute the sum of mx(L, R) * mn(L, R) over all N * (N + 1) / 2 subarrays [L, R], modulo 10^9 + 7.

Input
The first line contains N.
The second line contains A_1 ... A_N.

Output
Print the sum of mx(L, R) * mn(L, R) over all 1 <= L <= R <= N, modulo 10^9 + 7.

Constraints
1 <= N <= 180000
1 <= A_i <= 10^9""",
        samples=[
            "4\n2 5 1 4\n",
        ],
        small_inputs=[
            "1\n1000000000\n",
            gen_60(400, 100, 6001),
            gen_60(2200, 10**9, 6002),
        ],
        large_inputs=[
            gen_60(180000, 10**9, 6011),
            gen_60(180000, 500, 6012),
            gen_60(180000, 10**9, 6013),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 1'000'000'007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    vector<long long> rmax(n), rmin(n), ps_max(n + 1), ps_min(n + 1), ps_prod(n + 1);
    long long ans = 0;

    function<void(int, int)> solve = [&](int lo, int hi) {
        if (lo == hi) {
            long long v = a[lo] % MOD;
            ans = (ans + v * v) % MOD;
            return;
        }
        int mid = (lo + hi) >> 1;
        solve(lo, mid);
        solve(mid + 1, hi);

        int rlen = hi - mid;
        ps_max[0] = ps_min[0] = ps_prod[0] = 0;
        long long cur_mx = 0, cur_mn = (1LL << 60);
        for (int j = mid + 1; j <= hi; j++) {
            cur_mx = max(cur_mx, a[j]);
            cur_mn = min(cur_mn, a[j]);
            int idx = j - mid; // 1 .. rlen
            rmax[idx - 1] = cur_mx;
            rmin[idx - 1] = cur_mn;
            long long mxm = cur_mx % MOD, mnm = cur_mn % MOD;
            ps_max[idx] = (ps_max[idx - 1] + mxm) % MOD;
            ps_min[idx] = (ps_min[idx - 1] + mnm) % MOD;
            ps_prod[idx] = (ps_prod[idx - 1] + mxm * mnm) % MOD;
        }

        long long lmx = 0, lmn = (1LL << 60);
        int p = 0, q = 0; // 0-indexed into rmax/rmin [0 .. rlen-1]
        for (int i = mid; i >= lo; i--) {
            lmx = max(lmx, a[i]);
            lmn = min(lmn, a[i]);
            while (p < rlen && rmax[p] <= lmx) p++;
            while (q < rlen && rmin[q] >= lmn) q++;

            long long lmx_m = lmx % MOD, lmn_m = lmn % MOD;
            int u = min(p, q), v = max(p, q);
            // [0, u): both lmx and lmn dominate
            long long ways = (long long)u * ((lmx_m * lmn_m) % MOD) % MOD;
            // [u, v): one side from left, one from right
            if (p < q) {
                long long s = (ps_max[q] - ps_max[p] + MOD) % MOD;
                ways = (ways + lmn_m * s) % MOD;
            } else if (q < p) {
                long long s = (ps_min[p] - ps_min[q] + MOD) % MOD;
                ways = (ways + lmx_m * s) % MOD;
            }
            // [v, rlen): both rmax and rmin dominate
            long long s2 = (ps_prod[rlen] - ps_prod[v] + MOD) % MOD;
            ways = (ways + s2) % MOD;
            ans += ways;
            if (ans >= MOD) ans -= MOD;
        }
    };

    solve(0, n - 1);
    cout << ans % MOD << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 1'000'000'007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long mx = a[i], mn = a[i];
        for (int j = i; j < n; j++) {
            if (a[j] > mx) mx = a[j];
            if (a[j] < mn) mn = a[j];
            ans = (ans + (mx % MOD) * (mn % MOD)) % MOD;
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_12())
