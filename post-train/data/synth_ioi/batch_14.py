"""Batch 14 (synth_66 .. synth_70): Math, Number Theory & Linear Algebra II — Subarray GCD Spectrum, Harmonic Floor Sum, Exact-GCD Triples Möbius/IE, Offset Lattice-Path Binomial Sum, Prefix XOR Basis Square Subsets."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


PRIMES_300 = [
    p for p in range(2, 301)
    if all(p % d != 0 for d in range(2, int(p ** 0.5) + 1))
]


def make_batch_14() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_66: GCD SPECTRUM (1900) - Subarray GCD Counts (CF 475D)
    # ------------------------------------------------------------------
    def gen_66(n: int, q: int, seed: int, rich_gcds: bool = False) -> str:
        r = random.Random(seed)
        if rich_gcds:
            # Multiply by 12 and build from small prime powers so gcd > 1 everywhere (no early-1 exit!)
            small_p = [2, 3, 5, 7, 11, 13]
            a = []
            for _ in range(n):
                v = 12
                for _ in range(18):
                    p = r.choice(small_p)
                    if v * p <= 10**9:
                        v *= p
                a.append(v)
        else:
            a = [r.randint(1, 10**9) for _ in range(n)]
        qs = []
        for _ in range(q):
            if r.random() < 0.7:
                idx = r.randrange(n)
                g = a[idx]
                for step in range(r.randint(0, 6)):
                    if idx + step < n:
                        import math
                        g = math.gcd(g, a[idx + step])
                qs.append(g)
            else:
                qs.append(r.randint(1, 1000))
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(map(str, qs)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_66",
        title="GCD Spectrum",
        rating=1900,
        tags=["number theory", "math", "binary search", "data structures"],
        notes="Ending at each index r, prefix-subarray GCDs take at most log2(max A) + 1 <= 31 distinct values; maintain compressed (gcd, length) list in O(N log^2 V).",
        story_and_spec="""You are given an integer array A_1, A_2, ..., A_N and Q queries. Each query gives a positive integer X and asks: for how many pairs (L, R) with 1 <= L <= R <= N is gcd(A_L, A_{L+1}, ..., A_R) equal to X?

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains a query integer X.

Output
For each query print the number of subarrays with GCD equal to X on its own line.

Constraints
1 <= N, Q <= 180000
1 <= A_i, X <= 10^9""",
        samples=[
            "5 4\n6 12 4 8 2\n6\n4\n2\n3\n",
        ],
        small_inputs=[
            "1 2\n10\n10\n5\n",
            gen_66(300, 300, 6601, rich_gcds=True),
            gen_66(1500, 1500, 6602),
        ],
        large_inputs=[
            gen_66(180000, 180000, 6611, rich_gcds=True),
            gen_66(180000, 180000, 6612),
            gen_66(180000, 180000, 6613, rich_gcds=True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    vector<pair<int, long long>> cur, nxt, all_g;
    all_g.reserve(n * 10);
    for (int x : a) {
        nxt.clear();
        for (auto [g, c] : cur) {
            int ng = std::gcd(g, x);
            if (!nxt.empty() && nxt.back().first == ng) nxt.back().second += c;
            else nxt.push_back({ng, c});
        }
        if (!nxt.empty() && nxt.back().first == x) nxt.back().second += 1;
        else nxt.push_back({x, 1});
        cur.swap(nxt);
        for (auto& pr : cur) all_g.push_back(pr);
    }

    sort(all_g.begin(), all_g.end());
    vector<pair<int, long long>> merged;
    merged.reserve(all_g.size());
    for (auto [g, c] : all_g) {
        if (!merged.empty() && merged.back().first == g) merged.back().second += c;
        else merged.push_back({g, c});
    }

    while (q--) {
        int x; cin >> x;
        auto it = lower_bound(merged.begin(), merged.end(), make_pair(x, -1LL));
        if (it != merged.end() && it->first == x) cout << it->second << '\n';
        else cout << 0 << '\n';
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
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    unordered_map<int, long long> cnt;
    cnt.reserve(4096);
    for (int i = 0; i < n; i++) {
        int g = 0;
        for (int j = i; j < n; j++) {
            g = std::gcd(g, a[j]);
            cnt[g]++;
        }
    }
    while (q--) {
        int x; cin >> x;
        auto it = cnt.find(x);
        cout << (it == cnt.end() ? 0LL : it->second) << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_67: FLOOR HARMONICS (2000) - Pairwise Sum of floor(A_i / A_j)
    # ------------------------------------------------------------------
    def gen_67(n: int, vmax: int, seed: int, skew_small: bool = False) -> str:
        r = random.Random(seed)
        if skew_small:
            a = [r.randint(1, 500) if r.random() < 0.5 else r.randint(1, vmax) for _ in range(n)]
        else:
            a = [r.randint(1, vmax) for _ in range(n)]
        return f"{n}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_67",
        title="Floor Harmonics",
        rating=2000,
        tags=["number theory", "math", "prefix sums", "harmonic series"],
        notes="Count frequencies up to V = 500,000 with prefix sums; for each distinct denominator d, iterate multiples k*d .. (k+1)*d - 1 in O(V log V) total.",
        story_and_spec="""You are given N positive integers A_1, A_2, ..., A_N. Compute the sum over ALL ordered pairs (i, j) with 1 <= i <= N and 1 <= j <= N of floor(A_i / A_j).

Input
The first line contains N.
The second line contains A_1 ... A_N.

Output
Print a single integer: the value of sum_{i=1..N} sum_{j=1..N} floor(A_i / A_j).

Constraints
1 <= N <= 200000
1 <= A_i <= 500000""",
        samples=[
            "4\n2 5 7 3\n",
        ],
        small_inputs=[
            "3\n1 1 1\n",
            gen_67(400, 1000, 6701),
            gen_67(1800, 500000, 6702, skew_small=True),
        ],
        large_inputs=[
            gen_67(200000, 500000, 6711),
            gen_67(200000, 500000, 6712, skew_small=True),
            gen_67(200000, 50000, 6713),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    int mx = 0;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        mx = max(mx, a[i]);
    }
    vector<int> cnt(mx + 1, 0), pref(mx + 1, 0);
    for (int x : a) cnt[x]++;
    for (int v = 1; v <= mx; v++) pref[v] = pref[v - 1] + cnt[v];

    long long ans = 0;
    for (int d = 1; d <= mx; d++) {
        if (!cnt[d]) continue;
        long long cd = cnt[d];
        for (int k = 1, lo = d; lo <= mx; k++, lo += d) {
            int hi = min(mx, lo + d - 1);
            long long num = pref[hi] - pref[lo - 1];
            ans += cd * k * num;
        }
    }
    cout << ans << '\n';
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
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    long long ans = 0;
    for (int j = 0; j < n; j++) {
        int d = a[j];
        for (int i = 0; i < n; i++) ans += a[i] / d;
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_68: EXACT GCD TRIPLES (2000) - Top-Down Multiples Inclusion-Exclusion
    # ------------------------------------------------------------------
    def gen_68(n: int, q: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = []
        for _ in range(n):
            base = r.choice([1, 2, 3, 4, 6, 12, 15, 24, 30, 60])
            a.append(base * r.randint(1, max(1, vmax // base)))
        qs = [r.randint(1, 120) if r.random() < 0.7 else r.randint(1, vmax) for _ in range(q)]
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(map(str, qs)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_68",
        title="Exact GCD Triples",
        rating=2000,
        tags=["number theory", "inclusion-exclusion", "combinatorics", "mobius"],
        notes="Count multiples mult[d] for d = 1..V, initialize ways[d] = C(mult[d], 3), and subtract ways[2d], ways[3d], ... from d = V down to 1 in O(V log V).",
        story_and_spec="""You are given N positive integers A_1, A_2, ..., A_N and Q queries. Each query specifies an integer G: count how many triples of indices (i, j, k) with 1 <= i < j < k <= N satisfy gcd(A_i, A_j, A_k) = G.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains G.

Output
For each query print the number of triples with GCD equal to G on its own line.

Constraints
3 <= N <= 200000
1 <= Q <= 200000
1 <= A_i, G <= 300000""",
        samples=[
            "6 4\n2 4 6 12 3 9\n1\n2\n3\n4\n",
        ],
        small_inputs=[
            "3 2\n6 10 15\n1\n2\n",
            gen_68(120, 100, 500, 6801),
            gen_68(210, 200, 300000, 6802),
        ],
        large_inputs=[
            gen_68(200000, 200000, 300000, 6811),
            gen_68(200000, 200000, 50000, 6812),
            gen_68(200000, 200000, 300000, 6813),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int V = 300000;
    vector<int> freq(V + 1, 0);
    int mx = 0;
    for (int i = 0; i < n; i++) {
        int x; cin >> x;
        freq[x]++;
        mx = max(mx, x);
    }

    vector<long long> ways(mx + 1, 0);
    for (int d = 1; d <= mx; d++) {
        long long c = 0;
        for (int m = d; m <= mx; m += d) c += freq[m];
        if (c >= 3) ways[d] = c * (c - 1) * (c - 2) / 6;
    }
    for (int d = mx; d >= 1; d--) {
        for (int m = d + d; m <= mx; m += d) ways[d] -= ways[m];
    }

    while (q--) {
        int g; cin >> g;
        cout << (g <= mx ? ways[g] : 0LL) << '\n';
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
    vector<int> a(n);
    int mx = 0;
    for (int i = 0; i < n; i++) { cin >> a[i]; mx = max(mx, a[i]); }
    vector<long long> ways(mx + 1, 0);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int g2 = std::gcd(a[i], a[j]);
            for (int k = j + 1; k < n; k++) {
                ways[std::gcd(g2, a[k])]++;
            }
        }
    }
    while (q--) {
        int g; cin >> g;
        cout << (g <= mx ? ways[g] : 0LL) << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_69: SKEWER COMBINATIONS (2100) - AGC001E Offset Grid Lattice Paths
    # ------------------------------------------------------------------
    def gen_69(n: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [str(n)] + [f"{r.randint(1, vmax)} {r.randint(1, vmax)}" for _ in range(n)]
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_69",
        title="Skewer Combinations",
        rating=2100,
        tags=["dp", "combinatorics", "lattice paths", "math"],
        notes="AGC001E BBQ Hard: C(A_i + A_j + B_i + B_j, A_i + A_j) counts lattice paths from (-A_i, -B_i) to (A_j, B_j); compute all pairs simultaneously on a 4005x4005 DP grid, subtract self-pairs, divide by 2.",
        story_and_spec="""A festival kitchen has N ingredient packs. Pack i contains A_i pieces of meat and B_i pieces of vegetables.

When two distinct packs i and j (1 <= i < j <= N) are combined onto a single skewer, the number of distinct ways to arrange all (A_i + A_j) indistinguishable meat pieces and (B_i + B_j) indistinguishable vegetable pieces in a row is the binomial coefficient:
C(A_i + A_j + B_i + B_j, A_i + A_j).

Compute the sum of C(A_i + A_j + B_i + B_j, A_i + A_j) over all N * (N - 1) / 2 pairs 1 <= i < j <= N, modulo 10^9 + 7.

Input
The first line contains N.
Each of the next N lines contains two integers A_i and B_i.

Output
Print the total sum modulo 10^9 + 7.

Constraints
2 <= N <= 200000
1 <= A_i, B_i <= 2000""",
        samples=[
            "3\n1 1\n1 2\n2 1\n",
        ],
        small_inputs=[
            "2\n2000 2000\n2000 2000\n",
            gen_69(300, 200, 6901),
            gen_69(1800, 2000, 6902),
        ],
        large_inputs=[
            gen_69(200000, 2000, 6911),
            gen_69(200000, 500, 6912),
            gen_69(200000, 2000, 6913),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;
static constexpr int INV2 = (MOD + 1) / 2;

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
    int n;
    cin >> n;
    vector<int> A(n), B(n);
    int mx = 0;
    for (int i = 0; i < n; i++) {
        cin >> A[i] >> B[i];
        mx = max({mx, A[i], B[i]});
    }

    const int OFF = mx + 1;
    const int LIM = 2 * OFF + 1;
    vector<int> dp(LIM * LIM, 0);
    auto idx = [&](int x, int y) { return x * LIM + y; };

    for (int i = 0; i < n; i++) {
        int p = idx(OFF - A[i], OFF - B[i]);
        dp[p]++;
        if (dp[p] >= MOD) dp[p] -= MOD;
    }

    for (int x = 1; x < LIM; x++) {
        int row = x * LIM, prow = (x - 1) * LIM;
        for (int y = 1; y < LIM; y++) {
            long long s = (long long)dp[row + y] + dp[prow + y] + dp[row + y - 1];
            dp[row + y] = (int)(s % MOD);
        }
    }

    int MAXF = 4 * mx + 5;
    vector<long long> fac(MAXF, 1), ifac(MAXF, 1);
    for (int i = 1; i < MAXF; i++) fac[i] = fac[i - 1] * i % MOD;
    ifac[MAXF - 1] = modpow(fac[MAXF - 1], MOD - 2);
    for (int i = MAXF - 2; i >= 0; i--) ifac[i] = ifac[i + 1] * (i + 1) % MOD;
    auto C = [&](int nn, int kk) -> long long {
        return fac[nn] * ifac[kk] % MOD * ifac[nn - kk] % MOD;
    };

    long long total = 0;
    for (int i = 0; i < n; i++) {
        total += dp[idx(OFF + A[i], OFF + B[i])];
        total -= C(2 * A[i] + 2 * B[i], 2 * A[i]);
        total %= MOD;
    }
    if (total < 0) total += MOD;
    cout << (total * INV2) % MOD << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

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
    int n;
    cin >> n;
    vector<int> A(n), B(n);
    for (int i = 0; i < n; i++) cin >> A[i] >> B[i];

    const int MAXF = 8010;
    vector<long long> fac(MAXF, 1), ifac(MAXF, 1);
    for (int i = 1; i < MAXF; i++) fac[i] = fac[i - 1] * i % MOD;
    ifac[MAXF - 1] = modpow(fac[MAXF - 1], MOD - 2);
    for (int i = MAXF - 2; i >= 0; i--) ifac[i] = ifac[i + 1] * (i + 1) % MOD;

    long long ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int x = A[i] + A[j];
            int y = B[i] + B[j];
            ans = (ans + fac[x + y] * ifac[x] % MOD * ifac[y]) % MOD;
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_70: SQUARE PRODUCT SUBSETS (2100) - Rightmost Prefix XOR Basis (62 Primes)
    # ------------------------------------------------------------------
    def gen_70(n: int, q: int, seed: int, max_p_idx: int = 62, wide_q: bool = False) -> str:
        r = random.Random(seed)
        pool = PRIMES_300[:max_p_idx]
        LIM = 10**18
        a = []
        for _ in range(n):
            v = 1
            steps = r.randint(0, 14)
            for _ in range(steps):
                p = r.choice(pool)
                if v <= LIM // p:
                    v *= p
            a.append(v)
        lines = [f"{n} {q}", " ".join(map(str, a))]
        for _ in range(q):
            if wide_q and r.random() < 0.85:
                l = r.randint(1, max(1, n // 5))
                rr = r.randint(max(l, 4 * n // 5), n)
            else:
                l, rr = r.randint(1, n), r.randint(1, n)
                if l > rr:
                    l, rr = rr, l
            lines.append(f"{l} {rr}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_70",
        title="Square Product Subsets",
        rating=2100,
        tags=["linear algebra", "bitmasks", "number theory", "offline queries", "xor basis"],
        notes="Every A_i is 300-smooth (<= 62 primes), mapping to a 62-bit parity mask; maintain a rightmost XOR basis sweeping R from 1 to N so dim(L..R) is the count of basis vectors with pos >= L, giving 2^((R-L+1) - dim) - 1.",
        story_and_spec="""You are given an array A_1, A_2, ..., A_N of positive integers such that EVERY prime factor of every A_i is at most 300 (note that A_i itself can be up to 10^18).

Answer Q queries. Each query specifies L and R (1 <= L <= R <= N): count how many NON-EMPTY subsets of indices S subseteq {L, L + 1, ..., R} have the property that the product of A_i over i in S is a perfect square, modulo 10^9 + 7.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L and R.

Output
For each query print the number of non-empty square-product subsets modulo 10^9 + 7 on its own line.

Constraints
1 <= N, Q <= 180000
1 <= A_i <= 10^18, and every prime factor of A_i is <= 300
1 <= L <= R <= N""",
        samples=[
            "5 4\n2 3 6 12 1\n1 3\n1 4\n2 5\n5 5\n",
        ],
        small_inputs=[
            "3 2\n4 9 25\n1 3\n2 2\n",
            gen_70(200, 250, 7001, max_p_idx=15),
            gen_70(900, 900, 7002, max_p_idx=62, wide_q=True),
        ],
        large_inputs=[
            gen_70(180000, 180000, 7011, max_p_idx=45, wide_q=True),
            gen_70(180000, 180000, 7012, max_p_idx=62, wide_q=True),
            gen_70(180000, 180000, 7013, max_p_idx=62, wide_q=False),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static constexpr int MOD = 1'000'000'007;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    vector<int> primes;
    for (int p = 2; p <= 300; p++) {
        bool ok = true;
        for (int d = 2; d * d <= p; d++) if (p % d == 0) ok = false;
        if (ok) primes.push_back(p);
    }
    const int P = (int)primes.size(); // 62

    int n, q;
    cin >> n >> q;
    vector<uint64_t> mask(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        unsigned long long x;
        cin >> x;
        uint64_t m = 0;
        for (int b = 0; b < P && x > 1; b++) {
            unsigned long long pr = primes[b];
            if (x % pr == 0) {
                int parity = 0;
                while (x % pr == 0) { x /= pr; parity ^= 1; }
                if (parity) m |= (1ULL << b);
            }
        }
        mask[i] = m;
    }

    vector<int> pow2(n + 1, 1);
    for (int i = 1; i <= n; i++) {
        pow2[i] = pow2[i - 1] * 2;
        if (pow2[i] >= MOD) pow2[i] -= MOD;
    }

    vector<vector<pair<int, int>>> qs(n + 1);
    for (int id = 0; id < q; id++) {
        int l, r;
        cin >> l >> r;
        qs[r].push_back({l, id});
    }

    uint64_t bvec[62] = {};
    int bpos[62] = {};
    vector<int> ans(q);

    for (int r = 1; r <= n; r++) {
        uint64_t x = mask[r];
        int pos = r;
        for (int b = P - 1; b >= 0; b--) {
            if (!((x >> b) & 1ULL)) continue;
            if (!bvec[b]) {
                bvec[b] = x;
                bpos[b] = pos;
                break;
            }
            if (bpos[b] < pos) {
                swap(bvec[b], x);
                swap(bpos[b], pos);
            }
            x ^= bvec[b];
        }
        for (auto [l, id] : qs[r]) {
            int dim = 0;
            for (int b = 0; b < P; b++) if (bpos[b] >= l) dim++;
            int free_vars = (r - l + 1) - dim;
            int ways = pow2[free_vars] - 1;
            if (ways < 0) ways += MOD;
            ans[id] = ways;
        }
    }

    for (int id = 0; id < q; id++) cout << ans[id] << '\n';
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

    vector<int> primes;
    for (int p = 2; p <= 300; p++) {
        bool ok = true;
        for (int d = 2; d * d <= p; d++) if (p % d == 0) ok = false;
        if (ok) primes.push_back(p);
    }
    const int P = (int)primes.size();

    int n, q;
    cin >> n >> q;
    vector<uint64_t> mask(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        unsigned long long x; cin >> x;
        uint64_t m = 0;
        for (int b = 0; b < P && x > 1; b++) {
            unsigned long long pr = primes[b];
            int parity = 0;
            while (x % pr == 0) { x /= pr; parity ^= 1; }
            if (parity) m |= (1ULL << b);
        }
        mask[i] = m;
    }

    vector<int> pow2(n + 1, 1);
    for (int i = 1; i <= n; i++) pow2[i] = (pow2[i - 1] * 2) % MOD;

    while (q--) {
        int l, r; cin >> l >> r;
        uint64_t basis[62] = {};
        int dim = 0;
        for (int i = l; i <= r; i++) {
            uint64_t x = mask[i];
            for (int b = P - 1; b >= 0 && x; b--) {
                if (!((x >> b) & 1ULL)) continue;
                if (!basis[b]) { basis[b] = x; dim++; break; }
                x ^= basis[b];
            }
        }
        int ways = pow2[(r - l + 1) - dim] - 1;
        if (ways < 0) ways += MOD;
        cout << ways << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_14())
