"""Batch 7 (synth_31 .. synth_35): Number Theory & Combinatorics — Mobius Toggle, Binary-Power Vector-Matrix Walks, Obstacle Grid Inclusion-Exclusion, NTT Convolution, Distinct Prime Sweep."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_07() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_31: HARMONIC CHIMES (1900) - Dynamic Coprime Pairs (Mobius)
    # ------------------------------------------------------------------
    def gen_31(n: int, q: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        qs = [str(r.randint(1, n)) for _ in range(q)]
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_31",
        title="Harmonic Chimes",
        rating=1900,
        tags=["number theory", "mobius inversion", "sieve", "inclusion-exclusion"],
        notes="Precompute squarefree divisors and mu(d) up to 500,000; toggling bell x updates cnt[d] for squarefree d | A_x and adjusts pair count by sum mu(d) * cnt[d].",
        story_and_spec="""A bell tower houses N chimes numbered 1 to N, where chime i has resonance integer A_i. Initially, all N chimes are silent (OFF).

You process Q toggle events. Each event gives an index X (1 <= X <= N): if chime X is currently OFF it turns ON, and if it is currently ON it turns OFF. After each event, report the number of unordered pairs of distinct currently-ON chimes {i, j} such that gcd(A_i, A_j) = 1.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains X.

Output
Print Q lines: after each toggle, the number of coprime active pairs.

Constraints
1 <= N, Q <= 180000
1 <= A_i <= 500000
1 <= X <= N""",
        samples=[
            "5 6\n2 3 4 6 1\n1\n2\n3\n4\n5\n2\n",
        ],
        expected_samples=["0\n1\n2\n2\n6\n3\n"],
        small_inputs=[
            "2 3\n1 1\n1\n2\n1\n",
            gen_31(400, 600, 1000, 3101),
            gen_31(1200, 1400, 500000, 3102),
        ],
        large_inputs=[
            gen_31(160000, 160000, 500000, 3111),
            gen_31(180000, 180000, 500000, 3112),
            gen_31(180000, 180000, 30030, 3113),
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
    int mx = 1;
    for (int i = 1; i <= n; i++) {
        cin >> a[i];
        mx = max(mx, a[i]);
    }
    vector<int> spf(mx + 1, 0), mu(mx + 1, 0);
    mu[1] = 1;
    vector<int> primes;
    for (int i = 2; i <= mx; i++) {
        if (!spf[i]) { spf[i] = i; primes.push_back(i); mu[i] = -1; }
        for (int p : primes) {
            if (1LL * i * p > mx) break;
            spf[i * p] = p;
            if (i % p == 0) { mu[i * p] = 0; break; }
            mu[i * p] = -mu[i];
        }
    }

    auto sqfree_divs = [&](int x, vector<int>& divs) {
        int ps[10], k = 0;
        while (x > 1) {
            int p = spf[x];
            ps[k++] = p;
            while (x % p == 0) x /= p;
        }
        divs.resize(1 << k);
        divs[0] = 1;
        for (int b = 0; b < k; b++) {
            int sz = 1 << b, p = ps[b];
            for (int m = 0; m < sz; m++) divs[sz + m] = divs[m] * p;
        }
    };

    vector<int> cnt(mx + 1, 0);
    vector<char> on(n + 1, 0);
    vector<int> divs;
    long long ans = 0;
    while (q--) {
        int idx;
        cin >> idx;
        sqfree_divs(a[idx], divs);
        if (!on[idx]) {
            on[idx] = 1;
            for (int d : divs) {
                ans += 1LL * mu[d] * cnt[d];
                cnt[d]++;
            }
        } else {
            on[idx] = 0;
            for (int d : divs) {
                cnt[d]--;
                ans -= 1LL * mu[d] * cnt[d];
            }
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
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<char> on(n + 1, 0);
    vector<int> active;
    long long ans = 0;
    while (q--) {
        int x;
        cin >> x;
        if (!on[x]) {
            for (int y : active) if (std::gcd(a[x], a[y]) == 1) ans++;
            on[x] = 1;
            active.push_back(x);
        } else {
            on[x] = 0;
            active.erase(find(active.begin(), active.end(), x));
            for (int y : active) if (std::gcd(a[x], a[y]) == 1) ans--;
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_32: CITADEL WALKS (2100) - Precomputed Matrix Powers + Row-Vector Mul
    # ------------------------------------------------------------------
    def gen_32(n: int, m: int, q: int, kmax: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {m} {q}"]
        for _ in range(m):
            lines.append(f"{r.randint(1, n)} {r.randint(1, n)}")
        for _ in range(q):
            lines.append(f"{r.randint(1, n)} {r.randint(1, n)} {r.randint(1, kmax)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_32",
        title="Citadel Walks",
        rating=2100,
        tags=["math", "matrices", "dp", "graphs"],
        notes="Precompute binary powers G^(2^b) for b=0..59 in O(60 * N^3), then answer each (U, V, K) query via 1xN vector * NxN matrix in O(60 * N^2).",
        story_and_spec="""A citadel has N gates numbered 1 to N and M directed corridors (self-loops and parallel corridors between the same pair of gates are allowed; distinct parallel corridors count as distinct steps).

You are given Q queries. Each query specifies a start gate U, an end gate V, and an integer step count K: find the number of distinct walks of length EXACTLY K corridors starting at U and ending at V, modulo 10^9 + 7.

Input
The first line contains N, M, and Q.
Each of the next M lines contains two integers A_i and B_i representing a directed corridor from A_i to B_i.
Each of the next Q lines contains U, V, and K.

Output
For each query print the number of walks modulo 10^9 + 7 on its own line.

Constraints
1 <= N <= 65
1 <= M <= 4000
1 <= Q <= 8000
1 <= U, V <= N
1 <= K <= 10^18""",
        samples=[
            "3 4 3\n1 2\n2 3\n1 3\n3 1\n1 3 1\n1 3 2\n1 1 3\n",
        ],
        expected_samples=["1\n1\n1\n"],
        small_inputs=[
            "1 2 2\n1 1\n1 1\n1 1 1\n1 1 5\n",
            gen_32(25, 120, 80, 200, 3201),
            gen_32(35, 200, 100, 250, 3202),
        ],
        large_inputs=[
            gen_32(60, 2500, 7000, 10**18, 3211),
            gen_32(65, 3000, 8000, 10**18, 3212),
            gen_32(65, 3500, 8000, 10**18, 3213),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;
const unsigned long long MOD2 = 16ULL * MOD * MOD;
using Mat = array<array<long long, 65>, 65>;

Mat mul(const Mat& A, const Mat& B, int n) {
    Mat C{};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            unsigned long long s = 0;
            for (int k = 0; k < n; k++) {
                s += (unsigned long long)A[i][k] * B[k][j];
                if (s >= MOD2) s -= MOD2;
            }
            C[i][j] = (long long)(s % MOD);
        }
    }
    return C;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    const int LOG = 60;
    vector<Mat> pw(LOG);
    for (auto& M : pw) for (auto& row : M) row.fill(0);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        pw[0][u - 1][v - 1] = (pw[0][u - 1][v - 1] + 1) % MOD;
    }
    for (int b = 1; b < LOG; b++) pw[b] = mul(pw[b - 1], pw[b - 1], n);

    array<long long, 65> vec, nvec;
    while (q--) {
        int u, v;
        long long k;
        cin >> u >> v >> k;
        vec.fill(0);
        vec[u - 1] = 1;
        for (int b = 0; b < LOG; b++) if ((k >> b) & 1LL) {
            const Mat& M = pw[b];
            for (int j = 0; j < n; j++) {
                unsigned long long s = 0;
                for (int x = 0; x < n; x++) {
                    s += (unsigned long long)vec[x] * M[x][j];
                    if (s >= MOD2) s -= MOD2;
                }
                nvec[j] = (long long)(s % MOD);
            }
            vec = nvec;
        }
        cout << vec[v - 1] << '\n';
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<long long>> g(n, vector<long long>(n, 0));
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        g[u - 1][v - 1] = (g[u - 1][v - 1] + 1) % MOD;
    }
    while (q--) {
        int u, v; long long k;
        cin >> u >> v >> k;
        vector<long long> dp(n, 0), ndp(n, 0);
        dp[u - 1] = 1;
        for (long long step = 0; step < k; step++) {
            fill(ndp.begin(), ndp.end(), 0);
            for (int i = 0; i < n; i++) if (dp[i]) {
                for (int j = 0; j < n; j++) if (g[i][j]) {
                    ndp[j] = (ndp[j] + dp[i] * g[i][j]) % MOD;
                }
            }
            dp.swap(ndp);
        }
        cout << dp[v - 1] << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_33: SUNKEN COURTYARD (2200) - Grid Paths with Obstacles (Inclusion-Exclusion)
    # ------------------------------------------------------------------
    def gen_33(h: int, w: int, k: int, seed: int) -> str:
        r = random.Random(seed)
        used = {(1, 1), (h, w)}
        obs = []
        while len(obs) < k:
            pt = (r.randint(1, h), r.randint(1, w))
            if pt not in used:
                used.add(pt)
                obs.append(pt)
        lines = [f"{h} {w} {k}"] + [f"{x} {y}" for x, y in obs]
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_33",
        title="Sunken Courtyard",
        rating=2200,
        tags=["combinatorics", "dp", "inclusion-exclusion"],
        notes="Sort obstacles + target (H, W) topologically; dp[i] = ways((1,1)->p_i) - sum_{p_j <= p_i} dp[j] * ways(p_j -> p_i) in O(H + W + K^2).",
        story_and_spec="""A courtyard is a grid of H rows and W columns (1-indexed from (1, 1) at top-left to (H, W) at bottom-right). K distinct cells other than (1, 1) and (H, W) are flooded and cannot be entered.

In each step you may move one cell down ((r, c) -> (r + 1, c)) or one cell right ((r, c) -> (r, c + 1)). Count the number of valid paths from (1, 1) to (H, W) that never step onto a flooded cell, modulo 10^9 + 7.

Input
The first line contains H, W, and K.
Each of the next K lines contains R_i and C_i, the coordinates of a flooded cell.

Output
Print a single integer: the number of valid paths modulo 10^9 + 7.

Constraints
2 <= H, W <= 200000
0 <= K <= 2500
1 <= R_i <= H, 1 <= C_i <= W
All flooded cells are distinct and neither (1, 1) nor (H, W) is flooded.""",
        samples=[
            "3 4 2\n2 2\n2 3\n",
        ],
        expected_samples=["2\n"],
        small_inputs=[
            "2 2 0\n",
            "3 3 2\n1 2\n2 1\n",
            gen_33(500, 600, 40, 3301),
            gen_33(1200, 1200, 150, 3302),
        ],
        large_inputs=[
            gen_33(180000, 180000, 2000, 3311),
            gen_33(200000, 200000, 2500, 3312),
            gen_33(200000, 190000, 2200, 3313),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

long long modpow(long long a, long long e) {
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
    int H, W, K;
    cin >> H >> W >> K;
    int LIM = H + W + 5;
    vector<long long> fact(LIM, 1), invf(LIM, 1);
    for (int i = 1; i < LIM; i++) fact[i] = fact[i - 1] * i % MOD;
    invf[LIM - 1] = modpow(fact[LIM - 1], MOD - 2);
    for (int i = LIM - 2; i >= 0; i--) invf[i] = invf[i + 1] * (i + 1) % MOD;

    auto ways = [&](int dr, int dc) -> long long {
        if (dr < 0 || dc < 0) return 0;
        return fact[dr + dc] * invf[dr] % MOD * invf[dc] % MOD;
    };

    vector<pair<int, int>> pts(K);
    for (int i = 0; i < K; i++) cin >> pts[i].first >> pts[i].second;
    pts.push_back({H, W});
    sort(pts.begin(), pts.end());

    int m = (int)pts.size();
    vector<long long> dp(m, 0);
    for (int i = 0; i < m; i++) {
        long long v = ways(pts[i].first - 1, pts[i].second - 1);
        for (int j = 0; j < i; j++) {
            if (pts[j].second <= pts[i].second) {
                v = (v - dp[j] * ways(pts[i].first - pts[j].first, pts[i].second - pts[j].second)) % MOD;
            }
        }
        if (v < 0) v += MOD;
        dp[i] = v;
    }
    cout << dp[m - 1] << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int H, W, K;
    cin >> H >> W >> K;
    vector<vector<char>> bad(H + 1, vector<char>(W + 1, 0));
    for (int i = 0; i < K; i++) {
        int r, c; cin >> r >> c;
        bad[r][c] = 1;
    }
    vector<long long> dp(W + 1, 0);
    dp[1] = 1;
    for (int r = 1; r <= H; r++) {
        for (int c = 1; c <= W; c++) {
            if (bad[r][c]) dp[c] = 0;
            else if (!(r == 1 && c == 1)) dp[c] = (dp[c] + dp[c - 1]) % MOD;
        }
    }
    cout << dp[W] << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_34: ARTISAN ALLOYS (2100) - NTT Polynomial Self-Convolution
    # ------------------------------------------------------------------
    def gen_34(n: int, q: int, lmax: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {q}"]
        for _ in range(n):
            lines.append(f"{r.randint(1, lmax)} {r.randint(1, 10**8)}")
        for _ in range(q):
            lines.append(str(r.randint(2, 2 * lmax)))
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_34",
        title="Artisan Alloys",
        rating=2100,
        tags=["math", "fft", "ntt", "combinatorics"],
        notes="Build polynomial P(x) = sum_i W_i x^{L_i}; compute P(x)^2 mod 998244353 via NTT, subtract diagonal sum_{2 L_i = S} W_i^2, and multiply by inv(2).",
        story_and_spec="""A royal foundry holds N ingots numbered 1 to N. Ingot i has mass M_i (1 <= M_i <= 200000) and purity grade W_i.

When two DISTINCT ingots i and j (1 <= i < j <= N) are melted together, they form an alloy bar of total mass M_i + M_j and value W_i * W_j. Given Q target masses S_1, ..., S_Q, compute for each target mass S the sum of W_i * W_j over all unordered pairs {i, j} (i < j) with M_i + M_j = S, modulo 998244353.

Input
The first line contains N and Q.
Each of the next N lines contains M_i and W_i.
Each of the next Q lines contains S.

Output
For each target mass S print the total alloy value modulo 998244353 on its own line.

Constraints
2 <= N <= 200000
1 <= Q <= 200000
1 <= M_i <= 200000
1 <= W_i <= 10^8
2 <= S <= 400000""",
        samples=[
            "4 4\n1 3\n2 5\n2 4\n4 2\n3\n4\n5\n7\n",
        ],
        expected_samples=["27\n20\n6\n0\n"],
        small_inputs=[
            "2 2\n5 10\n5 20\n10\n9\n",
            gen_34(500, 500, 1000, 3401),
            gen_34(1600, 1600, 200000, 3402),
        ],
        large_inputs=[
            gen_34(180000, 180000, 200000, 3411),
            gen_34(200000, 200000, 200000, 3412),
            gen_34(200000, 200000, 150000, 3413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353LL;
const long long G = 3LL;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e > 0) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

void ntt(vector<long long>& a, bool invert) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        long long wlen = modpow(G, (MOD - 1) / len);
        if (invert) wlen = modpow(wlen, MOD - 2);
        for (int i = 0; i < n; i += len) {
            long long w = 1;
            int half = len >> 1;
            for (int j = 0; j < half; j++) {
                long long u = a[i + j];
                long long v = a[i + j + half] * w % MOD;
                a[i + j] = u + v < MOD ? u + v : u + v - MOD;
                a[i + j + half] = u - v >= 0 ? u - v : u - v + MOD;
                w = w * wlen % MOD;
            }
        }
    }
    if (invert) {
        long long inv_n = modpow(n, MOD - 2);
        for (long long& x : a) x = x * inv_n % MOD;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int MAXM = 200000;
    int sz = 1;
    while (sz <= 2 * MAXM + 2) sz <<= 1;
    vector<long long> poly(sz, 0), diag(2 * MAXM + 2, 0);
    for (int i = 0; i < n; i++) {
        int m; long long w;
        cin >> m >> w;
        w %= MOD;
        poly[m] = (poly[m] + w) % MOD;
        diag[2 * m] = (diag[2 * m] + w * w) % MOD;
    }
    ntt(poly, false);
    for (long long& x : poly) x = x * x % MOD;
    ntt(poly, true);
    long long inv2 = (MOD + 1) / 2;
    while (q--) {
        int s;
        cin >> s;
        long long ans = 0;
        if (s <= 2 * MAXM) {
            ans = (poly[s] - diag[s] + MOD) % MOD * inv2 % MOD;
        }
        cout << ans << '\n';
    }
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 998244353LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> m(n);
    vector<long long> w(n);
    for (int i = 0; i < n; i++) { cin >> m[i] >> w[i]; w[i] %= MOD; }
    vector<long long> ans(400005, 0);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int s = m[i] + m[j];
            ans[s] = (ans[s] + w[i] * w[j]) % MOD;
        }
    }
    while (q--) {
        int s; cin >> s;
        cout << ans[s] << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_35: PRIME LEDGER (1800) - Sieve + Offline Distinct Prime Sweep
    # ------------------------------------------------------------------
    def gen_35(n: int, q: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        qs = []
        for _ in range(q):
            L, R = r.randint(1, n), r.randint(1, n)
            if L > R:
                L, R = R, L
            qs.append(f"{L} {R}")
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_35",
        title="Prime Ledger",
        rating=1800,
        tags=["number theory", "sieve", "fenwick tree", "offline"],
        notes="Factorize each A_i <= 10^6 into <= 7 distinct primes via linear SPF sieve, then run offline R-sweep + Fenwick tree tracking last occurrence of each prime.",
        story_and_spec="""A ledger lists N positive integers A_1, A_2, ..., A_N. You are asked Q range questions: each question specifies L and R, and asks how many DISTINCT prime numbers divide the product A_L * A_{L+1} * ... * A_R.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L and R.

Output
For each question print the count of distinct prime divisors on its own line.

Constraints
1 <= N, Q <= 200000
1 <= A_i <= 1000000
1 <= L <= R <= N""",
        samples=[
            "5 4\n6 10 1 9 7\n1 2\n2 4\n3 3\n1 5\n",
        ],
        expected_samples=["3\n3\n0\n4\n"],
        small_inputs=[
            "3 2\n1 1 1\n1 3\n2 2\n",
            gen_35(400, 500, 1000, 3501),
            gen_35(1400, 1500, 1000000, 3502),
        ],
        large_inputs=[
            gen_35(180000, 180000, 1000000, 3511),
            gen_35(200000, 200000, 1000000, 3512),
            gen_35(200000, 200000, 510510, 3513),
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
    int mx = 2;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }

    vector<int> spf(mx + 1, 0), primes;
    for (int i = 2; i <= mx; i++) {
        if (!spf[i]) { spf[i] = i; primes.push_back(i); }
        for (int p : primes) {
            if (1LL * i * p > mx) break;
            spf[i * p] = p;
            if (i % p == 0) break;
        }
    }

    vector<vector<pair<int, int>>> by_r(n + 1);
    for (int i = 0; i < q; i++) {
        int l, r; cin >> l >> r;
        by_r[r].push_back({l, i});
    }

    vector<int> bit(n + 2, 0), last_pos(mx + 1, 0), ans(q, 0);
    auto add = [&](int idx, int d) {
        for (; idx <= n; idx += idx & -idx) bit[idx] += d;
    };
    auto sum = [&](int idx) {
        int s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    int total_active = 0;
    for (int r = 1; r <= n; r++) {
        int x = a[r];
        while (x > 1) {
            int p = spf[x];
            while (x % p == 0) x /= p;
            if (last_pos[p]) add(last_pos[p], -1);
            else total_active++;
            last_pos[p] = r;
            add(r, +1);
        }
        for (auto [l, id] : by_r[r]) {
            ans[id] = total_active - sum(l - 1);
        }
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
    int n, q;
    cin >> n >> q;
    vector<int> a(n + 1);
    int mx = 2;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }
    vector<int> spf(mx + 1, 0), primes;
    for (int i = 2; i <= mx; i++) {
        if (!spf[i]) { spf[i] = i; primes.push_back(i); }
        for (int p : primes) {
            if (1LL * i * p > mx) break;
            spf[i * p] = p;
            if (i % p == 0) break;
        }
    }
    vector<int> seen(mx + 1, 0);
    for (int step = 1; step <= q; step++) {
        int l, r; cin >> l >> r;
        int cnt = 0;
        for (int i = l; i <= r; i++) {
            int x = a[i];
            while (x > 1) {
                int p = spf[x];
                while (x % p == 0) x /= p;
                if (seen[p] != step) { seen[p] = step; cnt++; }
            }
        }
        cout << cnt << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_07())
