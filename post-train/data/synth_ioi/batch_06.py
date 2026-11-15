"""Batch 6 (synth_26 .. synth_30): Greedy, Interval Lifting, Regret Heap, Slope Trick, Fractional Binary Search, Parallel Binary Search."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_06() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_26: GUARD ROTATIONS (1800) - Interval Cover Jumps via Binary Lifting
    # ------------------------------------------------------------------
    def gen_26(k: int, m: int, q: int, max_len: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{k} {m} {q}"]
        for _ in range(m):
            l = r.randint(1, k)
            rr = min(k, l + r.randint(0, max_len))
            lines.append(f"{l} {rr}")
        for _ in range(q):
            a, b = r.randint(1, k), r.randint(1, k)
            if a > b:
                a, b = b, a
            if r.random() < 0.65:
                a = r.randint(1, max(1, k // 5))
                b = r.randint(max(a, 4 * k // 5), k)
            lines.append(f"{a} {b}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_26",
        title="Guard Rotations",
        rating=1800,
        tags=["greedy", "binary lifting", "intervals"],
        notes="Precompute best_r[x] = max_{L_i <= x} (R_i + 1) via prefix max, then binary-lift jumps from A until position > B.",
        story_and_spec="""A border watch runs for K integer hours numbered 1 to K. There are M available guard shifts; shift i covers every integer hour from L_i to R_i inclusive.

You are given Q inspection queries. Query j specifies A and B (1 <= A <= B <= K): find the minimum number of shifts needed so that every integer hour from A to B inclusive is covered by at least one chosen shift, or print -1 if it is impossible.

Input
The first line contains K, M, and Q.
Each of the next M lines contains L_i and R_i.
Each of the next Q lines contains A and B.

Output
For each query print the minimum number of shifts (or -1) on its own line.

Constraints
1 <= K, M, Q <= 200000
1 <= L_i <= R_i <= K
1 <= A <= B <= K""",
        samples=[
            "10 5 4\n1 4\n3 7\n6 9\n9 10\n2 5\n1 10\n2 7\n5 8\n8 10\n",
        ],
        small_inputs=[
            "5 2 3\n1 2\n4 5\n1 2\n1 5\n3 3\n",
            gen_26(500, 600, 500, 15, 2601),
            gen_26(1600, 1800, 1600, 20, 2602),
        ],
        large_inputs=[
            # Short intervals force O(K) jumps per query without binary lifting
            (lambda: (
                lambda k, q, r: (
                    f"{k} {k} {q}\n"
                    + "\n".join(f"{i} {min(k, i + r.randint(1, 3))}" for i in range(1, k + 1)) + "\n"
                    + "\n".join(f"{r.randint(1, k // 8)} {r.randint(7 * k // 8, k)}" for _ in range(q)) + "\n"
                )
            )(190000, 190000, random.Random(2611)))(),
            gen_26(200000, 200000, 200000, 8, 2612),
            gen_26(200000, 200000, 200000, 25, 2613),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int K, m, q;
    cin >> K >> m >> q;
    vector<int> reach(K + 2, 0);
    for (int i = 0; i < m; i++) {
        int l, r;
        cin >> l >> r;
        reach[l] = max(reach[l], r + 1);
    }
    for (int x = 1; x <= K + 1; x++) {
        reach[x] = max({reach[x], reach[x - 1], x});
    }
    int LOG = 1;
    while ((1 << LOG) <= K + 2) LOG++;
    vector<vector<int>> up(LOG, vector<int>(K + 2, K + 1));
    for (int x = 1; x <= K + 1; x++) up[0][x] = reach[x];
    for (int b = 1; b < LOG; b++)
        for (int x = 1; x <= K + 1; x++)
            up[b][x] = up[b - 1][up[b - 1][x]];

    while (q--) {
        int a, b;
        cin >> a >> b;
        int target = b + 1;
        if (up[LOG - 1][a] < target) {
            cout << -1 << '\n';
            continue;
        }
        int ans = 0, cur = a;
        for (int k = LOG - 1; k >= 0; k--) {
            if (up[k][cur] < target) {
                ans += (1 << k);
                cur = up[k][cur];
            }
        }
        cout << (ans + 1) << '\n';
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
    int K, m, q;
    cin >> K >> m >> q;
    vector<int> reach(K + 2, 0);
    for (int i = 0; i < m; i++) {
        int l, r;
        cin >> l >> r;
        reach[l] = max(reach[l], r + 1);
    }
    for (int x = 1; x <= K + 1; x++) reach[x] = max(reach[x], reach[x - 1]);
    while (q--) {
        int a, b;
        cin >> a >> b;
        int cur = a, steps = 0;
        while (cur <= b) {
            if (reach[cur] <= cur) { steps = -1; break; }
            cur = reach[cur];
            steps++;
        }
        cout << steps << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_27: ALCHEMIST VIALS (1900) - Online Regret-Heap Greedy (All Prefixes)
    # ------------------------------------------------------------------
    def gen_27(n: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(-10**9, 5 * 10**8) for _ in range(n)]
        return f"{n}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_27",
        title="Alchemist Vials",
        rating=1900,
        tags=["greedy", "priority queue", "dp"],
        notes="Min-heap regret greedy: tentatively take every vial; whenever running sum < 0, eject the smallest (most negative) taken vial. Optimal for every prefix 1..i simultaneously.",
        story_and_spec="""N alchemical vials sit on a shelf numbered 1 to N from left to right. Vial i changes an alchemist's vitality by A_i (which may be positive, zero, or negative).

Starting with vitality 0 and walking from vial 1 to vial k, the alchemist may choose to drink or skip each vial in order, subject to the rule that vitality must NEVER drop below 0 at any moment.

For EVERY prefix k = 1, 2, ..., N, print the maximum number of vials the alchemist can drink from the first k vials.

Input
The first line contains N.
The second line contains A_1 ... A_N.

Output
Print N space-separated integers on a single line: the k-th integer is the answer for prefix 1..k.

Constraints
1 <= N <= 200000
-10^9 <= A_i <= 10^9""",
        samples=[
            "6\n4 -4 1 -3 1 -3\n",
        ],
        expected_samples=["1 2 3 3 4 5\n"],
        small_inputs=[
            "4\n-5 -1 0 -2\n",
            gen_27(400, 2701),
            gen_27(1600, 2702),
            gen_27(2000, 2703),
        ],
        large_inputs=[
            gen_27(180000, 2711),
            gen_27(200000, 2712),
            gen_27(200000, 2713),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    priority_queue<long long, vector<long long>, greater<>> pq;
    long long sum = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        sum += x;
        pq.push(x);
        if (sum < 0) {
            sum -= pq.top();
            pq.pop();
        }
        cout << pq.size() << (i == n ? '\n' : ' ');
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
    cin >> n;
    const long long NEG = -(1LL << 60);
    vector<long long> dp(n + 1, NEG);
    dp[0] = 0;
    int best = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        for (int k = best + 1; k >= 1; k--) {
            if (dp[k - 1] != NEG && dp[k - 1] + x >= 0) {
                dp[k] = max(dp[k], dp[k - 1] + x);
                if (k > best) best = k;
            }
        }
        cout << best << (i == n ? '\n' : ' ');
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_28: STEPPED TERRACES (2200) - Slope Trick (Strictly Increasing Array)
    # ------------------------------------------------------------------
    def gen_28(n: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        return f"{n}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_28",
        title="Stepped Terraces",
        rating=2200,
        tags=["dp", "slope trick", "priority queue", "convexity"],
        notes="Substitute B'_i = B_i - i (can be any integer) and A'_i = A_i - i to reduce to non-decreasing L1 regression; solve via max-heap Slope Trick in O(N log N).",
        story_and_spec="""A hillside has N garden terraces numbered 1 to N, where terrace i currently has integer elevation A_i. Raising or lowering any terrace by 1 metre costs 1 unit of labour.

Choose new integer elevations B_1, B_2, ..., B_N (negative or zero integer elevations are allowed) such that B_1 < B_2 < ... < B_N (strictly increasing), minimizing the total labour sum_{i=1..N} |A_i - B_i|.

Input
The first line contains N.
The second line contains A_1 ... A_N.

Output
Print a single integer: the minimum total labour required.

Constraints
1 <= N <= 200000
1 <= A_i <= 10^9""",
        samples=[
            "5\n5 4 3 2 1\n",
            "4\n2 2 2 2\n",
        ],
        small_inputs=[
            "1\n100\n",
            gen_28(400, 1000, 2801),
            gen_28(1500, 10**9, 2802),
            gen_28(1800, 50, 2803),
        ],
        large_inputs=[
            gen_28(180000, 10**9, 2811),
            gen_28(200000, 10**9, 2812),
            (lambda: f"200000\n" + " ".join(str(200000 - i) for i in range(200000)) + "\n")(),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    priority_queue<long long> pq;
    long long ans = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        x -= i;
        pq.push(x);
        pq.push(x);
        ans += pq.top() - x;
        pq.pop();
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
    vector<long long> a(n), vals;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        a[i] -= (i + 1);
        vals.push_back(a[i]);
    }
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    vector<long long> dp(m, 0);
    for (int i = 0; i < n; i++) {
        long long best = (1LL << 62);
        for (int j = 0; j < m; j++) {
            best = min(best, dp[j]);
            dp[j] = best + llabs(a[i] - vals[j]);
        }
    }
    cout << *min_element(dp.begin(), dp.end()) << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_29: ORE ASSAY (2000) - 0/1 Fractional Binary Search + Prefix Min
    # ------------------------------------------------------------------
    def gen_29(n: int, k: int, seed: int) -> str:
        r = random.Random(seed)
        g = [r.randint(0, 10**6) for _ in range(n)]
        rk = [r.randint(1, 10**4) for _ in range(n)]
        return f"{n} {k}\n" + " ".join(map(str, g)) + "\n" + " ".join(map(str, rk)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_29",
        title="Ore Assay",
        rating=2000,
        tags=["binary search", "prefix sums", "fractional programming"],
        notes="Binary search integer X in [0, 10^12] testing whether max_{r-l+1>=K} sum(10^6 * G_i - X * R_i) >= 0 via prefix minimums.",
        story_and_spec="""A mine tunnel exposes N consecutive ore veins numbered 1 to N. Vein i contains G_i milligrams of gold embedded in R_i kilograms of rock. Mining regulations require excavating a single contiguous block of AT LEAST K consecutive veins.

If veins L, L+1, ..., R (with R - L + 1 >= K) are excavated, the yield density is (G_L + ... + G_R) / (R_L + ... + R_R). Print floor(10^6 * M), where M is the maximum achievable yield density over all valid blocks.

Input
The first line contains N and K.
The second line contains G_1 ... G_N.
The third line contains R_1 ... R_N.

Output
Print a single integer: floor(10^6 * M).

Constraints
1 <= K <= N <= 200000
0 <= G_i <= 10^6
1 <= R_i <= 10^4""",
        samples=[
            "5 2\n10 2 20 1 15\n5 5 4 5 3\n",
        ],
        small_inputs=[
            "3 3\n0 0 0\n1 2 3\n",
            gen_29(400, 15, 2901),
            gen_29(1500, 100, 2902),
            gen_29(1800, 5, 2903),
        ],
        large_inputs=[
            gen_29(180000, 5000, 2911),
            gen_29(200000, 30000, 2912),
            gen_29(200000, 50, 2913),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> g(n + 1), r(n + 1);
    for (int i = 1; i <= n; i++) cin >> g[i];
    for (int i = 1; i <= n; i++) cin >> r[i];

    vector<__int128> pref(n + 1, 0);
    auto ok = [&](long long X) -> bool {
        __int128 mn = 0;
        for (int i = 1; i <= n; i++) {
            pref[i] = pref[i - 1] + (__int128)1000000LL * g[i] - (__int128)X * r[i];
            if (i >= k) {
                if (pref[i - k] < mn) mn = pref[i - k];
                if (pref[i] - mn >= 0) return true;
            }
        }
        return false;
    };

    long long lo = 0, hi = 1000000000000LL;
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (ok(mid)) lo = mid;
        else hi = mid - 1;
    }
    cout << lo << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> pg(n + 1, 0), pr(n + 1, 0);
    for (int i = 1; i <= n; i++) { long long x; cin >> x; pg[i] = pg[i - 1] + x; }
    for (int i = 1; i <= n; i++) { long long x; cin >> x; pr[i] = pr[i - 1] + x; }

    long long best = 0;
    for (int l = 1; l + k - 1 <= n; l++) {
        for (int r = l + k - 1; r <= n; r++) {
            __int128 num = (__int128)1000000LL * (pg[r] - pg[l - 1]);
            long long den = pr[r] - pr[l - 1];
            best = max(best, (long long)(num / den));
        }
    }
    cout << best << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_30: TIMBER QUOTAS (2300) - Parallel Binary Search (IOI Meteors Style)
    # ------------------------------------------------------------------
    def gen_30(n: int, m: int, q: int, seed: int) -> str:
        r = random.Random(seed)
        owners = [r.randint(1, n) for _ in range(m)]
        req = [r.randint(1, 10**11) for _ in range(n)]
        lines = [f"{n} {m} {q}", " ".join(map(str, owners)), " ".join(map(str, req))]
        for _ in range(q):
            l, rr = r.randint(1, m), r.randint(1, m)
            if l > rr:
                l, rr = rr, l
            lines.append(f"{l} {rr} {r.randint(1, 10**7)}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_30",
        title="Timber Quotas",
        rating=2300,
        tags=["binary search", "parallel binary search", "fenwick tree", "divide and conquer"],
        notes="Parallel binary search over Q days for all N provinces simultaneously, using a range-add point-query Fenwick tree capped at 10^15 to avoid overflow.",
        story_and_spec="""A kingdom has N provinces numbered 1 to N and M timber yards arranged in a row numbered 1 to M. Yard j belongs to province O_j. Province i needs to accumulate at least R_i logs in total across all timber yards it owns.

Over Q days (numbered 1 to Q), day t delivers W_t logs to EVERY yard j with L_t <= j <= R_t. For each province i = 1, 2, ..., N, find the earliest day t after which province i has accumulated at least R_i logs in total, or print -1 if its quota is still unmet after all Q days.

Input
The first line contains N, M, and Q.
The second line contains O_1 ... O_M.
The third line contains R_1 ... R_N.
Each of the next Q lines contains L_t, R_t, and W_t.

Output
Print N lines: on line i print the earliest day province i meets its quota, or -1 if never.

Constraints
1 <= N, M, Q <= 150000
1 <= O_j <= N
1 <= R_i <= 10^12
1 <= L_t <= R_t <= M
1 <= W_t <= 10^7""",
        samples=[
            "3 5 4\n1 2 1 3 2\n10 15 8\n1 3 4\n2 5 6\n1 5 5\n4 4 10\n",
        ],
        small_inputs=[
            "2 2 2\n1 2\n5 100\n1 2 5\n1 2 5\n",
            gen_30(300, 500, 500, 3001),
            gen_30(900, 1200, 1200, 3002),
        ],
        large_inputs=[
            gen_30(130000, 140000, 140000, 3011),
            gen_30(150000, 150000, 150000, 3012),
            gen_30(150000, 150000, 150000, 3013),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct RangeBIT {
    int n;
    vector<long long> t;
    RangeBIT(int n) : n(n), t(n + 2, 0) {}
    void reset() { fill(t.begin(), t.end(), 0); }
    void add_pt(int i, long long v) {
        for (; i <= n + 1; i += i & -i) t[i] += v;
    }
    void add_range(int l, int r, long long v) {
        add_pt(l, v);
        add_pt(r + 1, -v);
    }
    long long get(int i) const {
        long long s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<int>> yards(n + 1);
    for (int j = 1; j <= m; j++) {
        int o; cin >> o;
        yards[o].push_back(j);
    }
    vector<long long> req(n + 1);
    for (int i = 1; i <= n; i++) cin >> req[i];
    struct Upd { int l, r; long long w; };
    vector<Upd> upds(q + 1);
    for (int t = 1; t <= q; t++) cin >> upds[t].l >> upds[t].r >> upds[t].w;

    vector<int> lo(n + 1, 1), hi(n + 1, q + 1);
    RangeBIT bit(m);
    vector<vector<int>> buckets(q + 2);

    while (true) {
        bool any = false;
        for (int t = 1; t <= q; t++) buckets[t].clear();
        for (int i = 1; i <= n; i++) {
            if (lo[i] < hi[i]) {
                any = true;
                buckets[(lo[i] + hi[i]) >> 1].push_back(i);
            }
        }
        if (!any) break;
        bit.reset();
        for (int t = 1; t <= q; t++) {
            bit.add_range(upds[t].l, upds[t].r, upds[t].w);
            for (int prov : buckets[t]) {
                __int128 sum = 0;
                for (int y : yards[prov]) {
                    sum += bit.get(y);
                    if (sum >= req[prov]) break;
                }
                if (sum >= req[prov]) hi[prov] = t;
                else lo[prov] = t + 1;
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << (lo[i] <= q ? lo[i] : -1) << '\n';
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
    vector<int> own(m + 1);
    for (int j = 1; j <= m; j++) cin >> own[j];
    vector<long long> req(n + 1), cur(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> req[i];
    vector<int> ans(n + 1, -1);
    for (int t = 1; t <= q; t++) {
        int l, r; long long w;
        cin >> l >> r >> w;
        for (int j = l; j <= r; j++) {
            int p = own[j];
            if (ans[p] == -1) {
                cur[p] += w;
                if (cur[p] >= req[p]) ans[p] = t;
            }
        }
    }
    for (int i = 1; i <= n; i++) cout << ans[i] << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_06())
