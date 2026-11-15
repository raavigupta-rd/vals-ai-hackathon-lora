"""Batch 15 (synth_71 .. synth_75): Sweepline, Permutations & Search II — Rectangle Union Area, Good Permutation Subarrays, Min-Max Segment Inversions, K-th & Sum of K Smallest Differences, Multi-Platform Interval Scheduling."""

from __future__ import annotations
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_15() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_71: SKYLINE SHADOWS (2000) - Area of Union of Rectangles
    # ------------------------------------------------------------------
    def gen_71(n: int, coord_max: int, seed: int, wide_x: bool = False) -> str:
        r = random.Random(seed)
        lines = [str(n)]
        for _ in range(n):
            if wide_x and r.random() < 0.8:
                x1 = r.randint(0, coord_max // 4)
                x2 = r.randint(max(x1 + 1, 3 * coord_max // 4), coord_max)
            else:
                x1 = r.randint(0, coord_max - 1)
                x2 = r.randint(x1 + 1, min(coord_max, x1 + r.randint(1, max(2, coord_max // 3))))
            y1 = r.randint(0, coord_max - 1)
            y2 = r.randint(y1 + 1, min(coord_max, y1 + r.randint(1, max(2, coord_max // 3))))
            lines.append(f"{x1} {y1} {x2} {y2}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_71",
        title="Skyline Shadows",
        rating=2000,
        tags=["geometry", "sweepline", "segment tree", "coordinate compression"],
        notes="X-sweepline over +/-1 vertical edge events with a cover-count Segment Tree over compressed Y-intervals computing active union length in O(N log N).",
        story_and_spec="""On a flat plain, N rectangular solar canopies cast shadows on the ground. Canopy i covers the axis-aligned rectangle [X_{1,i}, X_{2,i}] x [Y_{1,i}, Y_{2,i}] (with X_{1,i} < X_{2,i} and Y_{1,i} < Y_{2,i}). Canopies may overlap arbitrarily.

Compute the exact total area of the union of all N rectangles.

Input
The first line contains N.
Each of the next N lines contains four integers X_{1,i}, Y_{1,i}, X_{2,i}, and Y_{2,i}.

Output
Print a single integer: the exact area of the union of the N rectangles.

Constraints
1 <= N <= 150000
0 <= X_{1,i} < X_{2,i} <= 10^9
0 <= Y_{1,i} < Y_{2,i} <= 10^9""",
        samples=[
            "3\n1 1 5 4\n3 2 7 6\n6 1 8 3\n",
        ],
        small_inputs=[
            "2\n0 0 1000000000 1000000000\n10 10 20 20\n",
            gen_71(120, 50, 7101),
            gen_71(300, 10**9, 7102, wide_x=True),
        ],
        large_inputs=[
            gen_71(140000, 10**9, 7111, wide_x=True),
            gen_71(150000, 10**9, 7112),
            gen_71(150000, 100000, 7113),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct Event {
    int x, y1, y2, delta;
    bool operator<(const Event& o) const { return x < o.x; }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<Event> evs;
    evs.reserve(2 * n);
    vector<int> ys;
    ys.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        evs.push_back({x1, y1, y2, +1});
        evs.push_back({x2, y1, y2, -1});
        ys.push_back(y1);
        ys.push_back(y2);
    }
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());
    sort(evs.begin(), evs.end());

    int m = (int)ys.size() - 1; // elementary intervals 0 .. m-1
    if (m <= 0) {
        cout << 0 << '\n';
        return 0;
    }

    vector<int> cnt(4 * m + 5, 0);
    vector<long long> cov(4 * m + 5, 0);

    function<void(int, int, int, int, int, int)> upd = [&](int u, int l, int r, int ql, int qr, int d) {
        if (ql <= l && r <= qr) {
            cnt[u] += d;
        } else {
            int mid = (l + r) >> 1;
            if (ql <= mid) upd(u << 1, l, mid, ql, qr, d);
            if (qr > mid) upd(u << 1 | 1, mid + 1, r, ql, qr, d);
        }
        if (cnt[u] > 0) cov[u] = (long long)ys[r + 1] - ys[l];
        else if (l == r) cov[u] = 0;
        else cov[u] = cov[u << 1] + cov[u << 1 | 1];
    };

    long long ans = 0;
    for (size_t i = 0; i < evs.size(); ) {
        int cur_x = evs[i].x;
        if (i > 0) {
            long long dx = (long long)cur_x - evs[i - 1].x;
            ans += dx * cov[1];
        }
        while (i < evs.size() && evs[i].x == cur_x) {
            int l = (int)(lower_bound(ys.begin(), ys.end(), evs[i].y1) - ys.begin());
            int r = (int)(lower_bound(ys.begin(), ys.end(), evs[i].y2) - ys.begin()) - 1;
            if (l <= r) upd(1, 0, m - 1, l, r, evs[i].delta);
            i++;
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
    struct Rect { int x1, y1, x2, y2; };
    vector<Rect> rs(n);
    vector<int> xs;
    xs.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        cin >> rs[i].x1 >> rs[i].y1 >> rs[i].x2 >> rs[i].y2;
        xs.push_back(rs[i].x1);
        xs.push_back(rs[i].x2);
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), ys_end := xs.end()), xs.end());

    long long ans = 0;
    vector<pair<int, int>> segs;
    segs.reserve(n);
    for (size_t i = 0; i + 1 < xs.size(); i++) {
        int xl = xs[i], xr = xs[i + 1];
        segs.clear();
        for (const auto& rc : rs) {
            if (rc.x1 <= xl && xr <= rc.x2) segs.push_back({rc.y1, rc.y2});
        }
        if (segs.empty()) continue;
        sort(segs.begin(), segs.end());
        long long ylen = 0;
        int cl = segs[0].first, cr = segs[0].second;
        for (size_t j = 1; j < segs.size(); j++) {
            if (segs[j].first <= cr) cr = max(cr, segs[j].second);
            else { ylen += cr - cl; cl = segs[j].first; cr = segs[j].second; }
        }
        ylen += cr - cl;
        ans += ylen * (xr - xl);
    }
    cout << ans << '\n';
    return 0;
}
""".replace("ys_end := xs.end()", "xs.end()"),
    ))

    # ------------------------------------------------------------------
    # synth_72: CONSECUTIVE BLOCKS (2200) - Good Subarrays of a Permutation (CF 526F)
    # ------------------------------------------------------------------
    def gen_72(n: int, seed: int, structured: bool = False) -> str:
        r = random.Random(seed)
        if structured:
            # Build nested blocks of consecutive integers so the answer is very large
            p = list(range(1, n + 1))
            b = 8
            while b <= n:
                for start in range(0, n - b + 1, b):
                    if r.random() < 0.7:
                        sub = p[start:start + b]
                        mid = b // 2
                        p[start:start + b] = sub[mid:] + sub[:mid]
                b *= 2
        else:
            p = list(range(1, n + 1))
            r.shuffle(p)
        return f"{n}\n" + " ".join(map(str, p)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_72",
        title="Consecutive Blocks",
        rating=2200,
        tags=["data structures", "segment tree", "monotonic stack", "permutations"],
        notes="CF 526F: in a permutation, max(L..R) - min(L..R) - (R - L) >= 0 always; sweep R maintaining (max - min + L) in a lazy (min_val, min_cnt) segment tree updated by monotonic stacks.",
        story_and_spec="""You are given a permutation P_1, P_2, ..., P_N of the integers {1, 2, ..., N}.

A contiguous subarray [L, R] (1 <= L <= R <= N) is called a CONSECUTIVE BLOCK if sorting the elements P_L, P_{L+1}, ..., P_R produces a sequence of consecutive integers (equivalently, max(P_L..P_R) - min(P_L..P_R) = R - L).

Count the total number of consecutive blocks [L, R].

Input
The first line contains N.
The second line contains P_1 ... P_N.

Output
Print a single integer: the number of pairs (L, R) with 1 <= L <= R <= N that form a consecutive block.

Constraints
1 <= N <= 200000
P_1 ... P_N is a permutation of {1, 2, ..., N}""",
        samples=[
            "5\n3 2 5 4 1\n",
            "4\n2 4 1 3\n",
        ],
        small_inputs=[
            "1\n1\n",
            gen_72(300, 7201, structured=True),
            gen_72(2000, 7202, structured=False),
        ],
        large_inputs=[
            gen_72(200000, 7211, structured=True),
            gen_72(200000, 7212, structured=False),
            gen_72(180000, 7213, structured=True),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    vector<int> mn, cnt, lz;
    SegTree(int n) : n(n), mn(4 * n + 5, 1e9), cnt(4 * n + 5, 0), lz(4 * n + 5, 0) {}

    void apply(int u, int v) {
        mn[u] += v;
        lz[u] += v;
    }
    void push(int u) {
        if (lz[u] != 0) {
            apply(u << 1, lz[u]);
            apply(u << 1 | 1, lz[u]);
            lz[u] = 0;
        }
    }
    void pull(int u) {
        int a = u << 1, b = u << 1 | 1;
        mn[u] = min(mn[a], mn[b]);
        cnt[u] = (mn[a] == mn[u] ? cnt[a] : 0) + (mn[b] == mn[u] ? cnt[b] : 0);
    }
    void activate(int u, int l, int r, int pos, int val) {
        if (l == r) {
            mn[u] = val;
            cnt[u] = 1;
            lz[u] = 0;
            return;
        }
        push(u);
        int mid = (l + r) >> 1;
        if (pos <= mid) activate(u << 1, l, mid, pos, val);
        else activate(u << 1 | 1, mid + 1, r, pos, val);
        pull(u);
    }
    void add(int u, int l, int r, int ql, int qr, int val) {
        if (ql <= l && r <= qr) {
            apply(u, val);
            return;
        }
        push(u);
        int mid = (l + r) >> 1;
        if (ql <= mid) add(u << 1, l, mid, ql, qr, val);
        if (qr > mid) add(u << 1 | 1, mid + 1, r, ql, qr, val);
        pull(u);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> p(n + 1);
    for (int i = 1; i <= n; i++) cin >> p[i];

    SegTree st(n);
    // Stack entries: (pos, val)
    vector<pair<int, int>> smx = {{0, INT_MAX}}, smn = {{0, INT_MIN}};
    long long ans = 0;

    for (int r = 1; r <= n; r++) {
        while (smx.back().second < p[r]) {
            int rpos = smx.back().first, old_v = smx.back().second;
            smx.pop_back();
            int lpos = smx.back().first + 1;
            st.add(1, 1, n, lpos, rpos, p[r] - old_v);
        }
        smx.push_back({r, p[r]});

        while (smn.back().second > p[r]) {
            int rpos = smn.back().first, old_v = smn.back().second;
            smn.pop_back();
            int lpos = smn.back().first + 1;
            st.add(1, 1, n, lpos, rpos, old_v - p[r]);
        }
        smn.push_back({r, p[r]});

        st.activate(1, 1, n, r, r);
        if (st.mn[1] == r) ans += st.cnt[1];
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
    vector<int> p(n);
    for (int i = 0; i < n; i++) cin >> p[i];
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        int mx = p[i], mn = p[i];
        for (int j = i; j < n; j++) {
            if (p[j] > mx) mx = p[j];
            if (p[j] < mn) mn = p[j];
            if (mx - mn == j - i) ans++;
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_73: INVERSION SLICES (2100) - Binary Search + Rollback Fenwick Partition
    # ------------------------------------------------------------------
    def gen_73(n: int, k: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        return f"{n} {k}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_73",
        title="Inversion Slices",
        rating=2100,
        tags=["binary search", "greedy", "fenwick tree", "data structures"],
        notes="Binary search the maximum allowed inversions X per segment; greedily extend the current segment with a Fenwick tree and selectively roll back only the current segment's elements when X is exceeded — O(N log N log(N^2)).",
        story_and_spec="""You are given an integer array A_1, A_2, ..., A_N and an integer K. You must partition the array into AT MOST K non-empty contiguous segments.

Within a single segment [L, R], an inversion is a pair of indices (i, j) with L <= i < j <= R and A_i > A_j.

Find the minimum possible value of M such that the array can be partitioned into at most K contiguous segments where EVERY segment contains at most M internal inversions.

Input
The first line contains N and K.
The second line contains A_1 ... A_N.

Output
Print a single integer: the minimum achievable maximum inversions in any segment.

Constraints
1 <= K <= N <= 100000
1 <= A_i <= 10^9""",
        samples=[
            "7 2\n5 1 4 2 6 3 1\n",
            "5 5\n9 7 5 3 1\n",
        ],
        small_inputs=[
            "4 1\n4 3 2 1\n",
            gen_73(250, 7, 50, 7301),
            gen_73(800, 4, 10**9, 7302),
        ],
        large_inputs=[
            gen_73(100000, 3, 10**9, 7311),
            gen_73(100000, 50, 10**9, 7312),
            gen_73(100000, 1500, 500, 7313),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    vector<int> vals = a;
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();
    vector<int> rk(n);
    for (int i = 0; i < n; i++) {
        rk[i] = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin()) + 1;
    }

    vector<int> bit(m + 1, 0);
    auto bit_add = [&](int idx, int d) {
        for (; idx <= m; idx += idx & -idx) bit[idx] += d;
    };
    auto bit_sum = [&](int idx) {
        int s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    auto ok = [&](long long lim) -> bool {
        int segs = 1;
        int seg_l = 0;
        long long inv = 0;
        for (int i = 0; i < n; i++) {
            int len = i - seg_l;
            int le = bit_sum(rk[i]);
            long long add = len - le;
            if (inv + add <= lim) {
                inv += add;
                bit_add(rk[i], +1);
            } else {
                for (int j = seg_l; j < i; j++) bit_add(rk[j], -1);
                segs++;
                if (segs > k) return false;
                seg_l = i;
                inv = 0;
                bit_add(rk[i], +1);
            }
        }
        for (int j = seg_l; j < n; j++) bit_add(rk[j], -1);
        return true;
    };

    long long lo = 0, hi = 1LL * n * (n - 1) / 2;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (ok(mid)) hi = mid;
        else lo = mid + 1;
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
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    auto ok = [&](long long lim) -> bool {
        int segs = 1, seg_l = 0;
        long long inv = 0;
        for (int i = 0; i < n; i++) {
            long long add = 0;
            for (int j = seg_l; j < i; j++) if (a[j] > a[i]) add++;
            if (inv + add <= lim) {
                inv += add;
            } else {
                segs++;
                if (segs > k) return false;
                seg_l = i;
                inv = 0;
            }
        }
        return true;
    };

    long long lo = 0, hi = 1LL * n * (n - 1) / 2;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (ok(mid)) hi = mid;
        else lo = mid + 1;
    }
    cout << lo << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_74: PAIRWISE DISTANCES (2000) - K-th Smallest & Sum of K Smallest |A_i - A_j|
    # ------------------------------------------------------------------
    def gen_74(n: int, vmax: int, k_frac: float, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(0, vmax) for _ in range(n)]
        tot = n * (n - 1) // 2
        k = max(1, min(tot, int(tot * k_frac)))
        return f"{n} {k}\n" + " ".join(map(str, a)) + "\n"

    specs.append(ProblemSpec(
        pid="synth_74",
        title="Pairwise Distances",
        rating=2000,
        tags=["binary search", "two pointers", "sorting", "prefix sums"],
        notes="Sort A, binary search the K-th difference D_K via two-pointer pair counting, then sum all differences strictly < D_K via prefix sums and add (K - cnt_lt) * D_K.",
        story_and_spec="""Given N integers A_1, A_2, ..., A_N, consider the multiset of all N * (N - 1) / 2 pairwise absolute differences |A_i - A_j| for 1 <= i < j <= N, sorted in non-decreasing order:
d_1 <= d_2 <= ... <= d_{N*(N-1)/2}.

Given K, find BOTH:
1. d_K, the K-th smallest pairwise absolute difference.
2. S_K = d_1 + d_2 + ... + d_K, the sum of the K smallest pairwise absolute differences.

Input
The first line contains N and K.
The second line contains A_1 ... A_N.

Output
Print two space-separated integers: d_K and S_K.

Constraints
2 <= N <= 200000
1 <= K <= N * (N - 1) / 2
0 <= A_i <= 10^8""",
        samples=[
            "4 4\n1 5 2 9\n",
            "4 3\n5 5 5 8\n",
        ],
        small_inputs=[
            "2 1\n0 100000000\n",
            gen_74(200, 100, 0.35, 7401),
            gen_74(900, 10**8, 0.72, 7402),
        ],
        large_inputs=[
            gen_74(200000, 10**8, 0.45, 7411),
            gen_74(200000, 5000, 0.80, 7412),
            gen_74(200000, 10**8, 0.01, 7413),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long K;
    cin >> n >> K;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    sort(a.begin(), a.end());

    vector<long long> pref(n + 1, 0);
    for (int i = 0; i < n; i++) pref[i + 1] = pref[i] + a[i];

    auto count_le = [&](long long D) -> long long {
        long long cnt = 0;
        int l = 0;
        for (int r = 0; r < n; r++) {
            while (a[r] - a[l] > D) l++;
            cnt += (r - l);
        }
        return cnt;
    };

    long long lo = 0, hi = a[n - 1] - a[0];
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_le(mid) >= K) hi = mid;
        else lo = mid + 1;
    }
    long long dK = lo;

    long long cnt_lt = 0, sum_lt = 0;
    if (dK > 0) {
        int l = 0;
        for (int r = 0; r < n; r++) {
            while (a[r] - a[l] >= dK) l++;
            long long c = r - l;
            cnt_lt += c;
            sum_lt += c * a[r] - (pref[r] - pref[l]);
        }
    }
    long long sK = sum_lt + (K - cnt_lt) * dK;
    cout << dK << ' ' << sK << '\n';
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
    long long K;
    cin >> n >> K;
    vector<long long> a(n);
    long long mn = (1LL << 60), mx = 0;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        mn = min(mn, a[i]);
        mx = max(mx, a[i]);
    }

    auto count_le = [&](long long D) -> long long {
        long long cnt = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (llabs(a[i] - a[j]) <= D) cnt++;
            }
        }
        return cnt;
    };

    long long lo = 0, hi = mx - mn;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_le(mid) >= K) hi = mid;
        else lo = mid + 1;
    }
    long long dK = lo, cnt_lt = 0, sum_lt = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            long long d = llabs(a[i] - a[j]);
            if (d < dK) {
                cnt_lt++;
                sum_lt += d;
            }
        }
    }
    cout << dK << ' ' << (sum_lt + (K - cnt_lt) * dK) << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_75: PLATFORM DISPATCH (1900) - K-Track Interval Packing (CSES Movie Festival II)
    # ------------------------------------------------------------------
    def gen_75(n: int, k: int, tmax: int, seed: int) -> str:
        r = random.Random(seed)
        lines = [f"{n} {k}"]
        for _ in range(n):
            a = r.randint(0, tmax - 1)
            b = r.randint(a + 1, min(tmax, a + r.randint(1, max(2, tmax // 4))))
            lines.append(f"{a} {b}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_75",
        title="Platform Dispatch",
        rating=1900,
        tags=["greedy", "sorting", "data structures", "multiset"],
        notes="Sort intervals by end time R_i ascending; assign each interval [L_i, R_i) to the available track with the GREATEST current end time <= L_i using multiset::upper_bound.",
        story_and_spec="""A central rail station has K parallel platforms and receives N train reservation requests. Train i requires a single platform from arrival time L_i to departure time R_i (with L_i < R_i).

A single platform cannot serve two overlapping trains, except back-to-back: if one train departs at time R_i and another arrives at time L_j >= R_i, they MAY share the same platform. Each accepted train must stay on one platform for its entire stay, and you may reject any subset of requests.

Find the maximum number of trains that can be accommodated across the K platforms.

Input
The first line contains N and K.
Each of the next N lines contains L_i and R_i.

Output
Print a single integer: the maximum number of trains that can be scheduled.

Constraints
1 <= K <= N <= 200000
0 <= L_i < R_i <= 10^9""",
        samples=[
            "5 2\n1 5\n6 8\n2 9\n1 10\n8 11\n",
        ],
        small_inputs=[
            "3 2\n1 5\n6 8\n2 9\n",
            gen_75(250, 20, 100, 7501),
            gen_75(1200, 350, 10**9, 7502),
        ],
        large_inputs=[
            gen_75(200000, 60000, 10**9, 7511),
            gen_75(200000, 5000, 10**9, 7512),
            gen_75(200000, 25000, 50000, 7513),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<pair<int, int>> iv(n);
    for (int i = 0; i < n; i++) cin >> iv[i].second >> iv[i].first; // (R, L)
    sort(iv.begin(), iv.end());

    multiset<int> ends;
    int ans = 0;
    for (auto [r, l] : iv) {
        auto it = ends.upper_bound(l);
        if (it != ends.begin()) {
            --it;
            ends.erase(it);
            ends.insert(r);
            ans++;
        } else if ((int)ends.size() < k) {
            ends.insert(r);
            ans++;
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
    int n, k;
    cin >> n >> k;
    vector<pair<int, int>> iv(n);
    for (int i = 0; i < n; i++) cin >> iv[i].second >> iv[i].first;
    sort(iv.begin(), iv.end());

    vector<int> ends(k, 0);
    int ans = 0;
    for (auto [r, l] : iv) {
        int best_idx = -1;
        for (int j = 0; j < k; j++) {
            if (ends[j] <= l && (best_idx == -1 || ends[j] > ends[best_idx])) {
                best_idx = j;
            }
        }
        if (best_idx != -1) {
            ends[best_idx] = r;
            ans++;
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_15())
