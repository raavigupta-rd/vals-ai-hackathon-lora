"""Batch 8 (synth_36 .. synth_40): Geometry, Convex Hull Extreme Queries, Polar Sweep, Persistent SegTree K-th Sum, Mo's Algorithm, DAG Sprague-Grundy Winning Moves."""

from __future__ import annotations
import math
import random
from builder import ProblemSpec, build_and_verify_batch


def make_batch_08() -> list[ProblemSpec]:
    specs: list[ProblemSpec] = []

    # ------------------------------------------------------------------
    # synth_36: SURVEYOR BEACONS (2000) - Convex Hull Extreme Dot Product
    # ------------------------------------------------------------------
    def gen_36(n: int, q: int, cmax: int, seed: int, hull_heavy: bool = False) -> str:
        r = random.Random(seed)
        pts = []
        if hull_heavy:
            # Points near a circle so convex hull has thousands of vertices!
            R = 5 * 10**8
            for i in range(n):
                ang = 2.0 * math.pi * i / n
                x = int(round(R * math.cos(ang)))
                y = int(round(R * math.sin(ang)))
                pts.append((x, y))
            r.shuffle(pts)
        else:
            pts = [(r.randint(-cmax, cmax), r.randint(-cmax, cmax)) for _ in range(n)]
        lines = [f"{n} {q}"] + [f"{x} {y}" for x, y in pts]
        for _ in range(q):
            a, b = 0, 0
            while a == 0 and b == 0:
                a, b = r.randint(-cmax, cmax), r.randint(-cmax, cmax)
            lines.append(f"{a} {b}")
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_36",
        title="Surveyor Beacons",
        rating=2000,
        tags=["geometry", "convex hull", "binary search"],
        notes="Build upper and lower convex hulls via monotone chain; answer max(A*x + B*y) by binary searching edge normal slopes on upper hull (B >= 0) or lower hull (B < 0).",
        story_and_spec="""A mountain range has N survey beacons, where beacon i is located at integer coordinates (X_i, Y_i). You are given Q directional queries: each query gives a non-zero integer vector (A, B) and asks for the maximum value of A * X_i + B * Y_i over all beacons i = 1, 2, ..., N.

Input
The first line contains N and Q.
Each of the next N lines contains X_i and Y_i.
Each of the next Q lines contains A and B (not both zero).

Output
For each query print the maximum dot product on its own line.

Constraints
1 <= N, Q <= 200000
-10^9 <= X_i, Y_i, A, B <= 10^9
(A, B) != (0, 0)""",
        samples=[
            "4 4\n1 2\n4 1\n-3 5\n0 -4\n2 3\n1 -1\n-1 0\n0 -2\n",
        ],
        small_inputs=[
            "1 2\n-5 7\n1 1\n-2 -3\n",
            gen_36(400, 500, 1000, 3601, False),
            gen_36(1500, 1500, 10**9, 3602, True),
        ],
        large_inputs=[
            gen_36(180000, 180000, 10**9, 3611, True),
            gen_36(200000, 200000, 10**9, 3612, True),
            gen_36(200000, 200000, 10**9, 3613, False),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct Pt {
    long long x, y;
    bool operator<(const Pt& o) const { return x != o.x ? x < o.x : y < o.y; }
    bool operator==(const Pt& o) const { return x == o.x && y == o.y; }
};

static inline __int128 cross(const Pt& a, const Pt& b, const Pt& c) {
    return (__int128)(b.x - a.x) * (c.y - a.y) - (__int128)(b.y - a.y) * (c.x - a.x);
}

static inline long long dot(const Pt& p, long long a, long long b) {
    return a * p.x + b * p.y;
}

long long max_on_chain(const vector<Pt>& h, long long a, long long b) {
    int lo = 0, hi = (int)h.size() - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (dot(h[mid], a, b) < dot(h[mid + 1], a, b)) lo = mid + 1;
        else hi = mid;
    }
    return max({dot(h.front(), a, b), dot(h.back(), a, b), dot(h[lo], a, b)});
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<Pt> pts(n);
    for (int i = 0; i < n; i++) cin >> pts[i].x >> pts[i].y;
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());

    vector<Pt> low, up;
    for (const auto& p : pts) {
        while (low.size() >= 2 && cross(low[low.size() - 2], low.back(), p) <= 0) low.pop_back();
        low.push_back(p);
        while (up.size() >= 2 && cross(up[up.size() - 2], up.back(), p) >= 0) up.pop_back();
        up.push_back(p);
    }

    while (q--) {
        long long a, b;
        cin >> a >> b;
        if (b == 0) {
            cout << max(dot(pts.front(), a, b), dot(pts.back(), a, b)) << '\n';
        } else if (b > 0) {
            cout << max_on_chain(up, a, b) << '\n';
        } else {
            cout << max_on_chain(low, a, b) << '\n';
        }
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
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    while (q--) {
        long long a, b;
        cin >> a >> b;
        long long best = -(1LL << 62);
        for (int i = 0; i < n; i++) best = max(best, a * x[i] + b * y[i]);
        cout << best << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_37: CITADEL ENCLOSURE (2100) - Polar Angle Two-Pointer Sweep
    # ------------------------------------------------------------------
    def gen_37(n: int, cmax: int, seed: int) -> str:
        r = random.Random(seed)
        dirs: set[tuple[int, int]] = set()
        pts: list[tuple[int, int]] = []
        while len(pts) < n:
            x = r.randint(-cmax, cmax)
            y = r.randint(-cmax, cmax)
            if x == 0 and y == 0:
                continue
            g = math.gcd(abs(x), abs(y))
            dx, dy = x // g, y // g
            if (dx, dy) in dirs or (-dx, -dy) in dirs:
                continue
            dirs.add((dx, dy))
            pts.append((x, y))
        return f"{n}\n" + "\n".join(f"{x} {y}" for x, y in pts) + "\n"

    specs.append(ProblemSpec(
        pid="synth_37",
        title="Citadel Enclosure",
        rating=2100,
        tags=["geometry", "radial sweep", "two pointers", "combinatorics"],
        notes="Sort points by exact integer polar angle around (0, 0); subtract from C(N, 3) all triples contained in a semicircle via two pointers.",
        story_and_spec="""A central citadel stands at the origin (0, 0). Around it stand N watchtowers; watchtower i is at integer coordinates (X_i, Y_i). No watchtower stands at (0, 0), and no line through (0, 0) passes through two or more watchtowers.

Count how many unordered triples of distinct watchtowers {i, j, k} form a triangle that strictly encloses the origin (0, 0).

Input
The first line contains N.
Each of the next N lines contains X_i and Y_i.

Output
Print a single integer: the number of enclosing triangles.

Constraints
3 <= N <= 180000
-10^9 <= X_i, Y_i <= 10^9
No line through (0, 0) contains two or more watchtowers.""",
        samples=[
            "4\n1 1\n-2 1\n0 -3\n2 1\n",
        ],
        expected_samples=["2\n"],
        small_inputs=[
            "3\n1 0\n-1 1\n-1 -1\n",
            "3\n1 1\n2 1\n1 2\n",
            gen_37(120, 5000, 3701),
            gen_37(260, 10**6, 3702),
        ],
        large_inputs=[
            gen_37(150000, 10**9, 3711),
            gen_37(180000, 10**9, 3712),
            gen_37(180000, 10**9, 3713),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

struct Pt {
    long long x, y;
    int half() const { return (y > 0 || (y == 0 && x > 0)) ? 0 : 1; }
};

static inline __int128 cross(const Pt& a, const Pt& b) {
    return (__int128)a.x * b.y - (__int128)a.y * b.x;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<Pt> p(n);
    for (int i = 0; i < n; i++) cin >> p[i].x >> p[i].y;
    sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
        int ha = a.half(), hb = b.half();
        if (ha != hb) return ha < hb;
        return cross(a, b) > 0;
    });
    p.resize(2 * n);
    for (int i = 0; i < n; i++) p[n + i] = p[i];

    long long total = (__int128)n * (n - 1) * (n - 2) / 6;
    long long bad = 0;
    int r = 0;
    for (int i = 0; i < n; i++) {
        if (r < i) r = i;
        while (r + 1 < i + n && cross(p[i], p[r + 1]) > 0) r++;
        long long m = r - i;
        bad += m * (m - 1) / 2;
    }
    cout << (total - bad) << '\n';
    return 0;
}
""",
        brute_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

static inline int sgn(__int128 v) { return (v > 0) - (v < 0); }

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    auto cr = [&](int i, int j) -> __int128 {
        return (__int128)x[i] * y[j] - (__int128)y[i] * x[j];
    };
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int s1 = sgn(cr(i, j));
            for (int k = j + 1; k < n; k++) {
                int s2 = sgn(cr(j, k));
                int s3 = sgn(cr(k, i));
                if (s1 == s2 && s2 == s3 && s1 != 0) ans++;
            }
        }
    }
    cout << ans << '\n';
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_38: RANKED GRANARIES (2000) - Persistent SegTree K-th & Sum of K Smallest
    # ------------------------------------------------------------------
    def gen_38(n: int, q: int, vmax: int, seed: int) -> str:
        r = random.Random(seed)
        a = [r.randint(1, vmax) for _ in range(n)]
        qs = []
        for _ in range(q):
            if r.random() < 0.7:
                L = r.randint(1, max(1, n // 5))
                R = r.randint(max(L, 4 * n // 5), n)
            else:
                L, R = r.randint(1, n), r.randint(1, n)
                if L > R:
                    L, R = R, L
            k = r.randint(1, R - L + 1)
            qs.append(f"{L} {R} {k}")
        return f"{n} {q}\n" + " ".join(map(str, a)) + "\n" + "\n".join(qs) + "\n"

    specs.append(ProblemSpec(
        pid="synth_38",
        title="Ranked Granaries",
        rating=2000,
        tags=["data structures", "persistent segment tree", "order statistics"],
        notes="Coordinate-compressed persistent segment tree storing (count, sum); walk down diff(root[R], root[L-1]) to get both K-th smallest and sum of K smallest.",
        story_and_spec="""A valley has N granaries in a row numbered 1 to N, where granary i stores A_i sacks of grain. You are asked Q audit queries. Each query specifies L, R, and K (1 <= K <= R - L + 1): if the values A_L, A_{L+1}, ..., A_R are sorted in non-decreasing order as B_1 <= B_2 <= ... <= B_{R-L+1}, report BOTH B_K (the K-th smallest value) AND the sum B_1 + B_2 + ... + B_K.

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L, R, and K.

Output
For each query print two space-separated integers on its own line: B_K and the sum of the K smallest values.

Constraints
1 <= N, Q <= 180000
1 <= A_i <= 10^9
1 <= L <= R <= N
1 <= K <= R - L + 1""",
        samples=[
            "6 4\n5 1 4 1 3 9\n1 6 3\n2 5 2\n1 4 4\n6 6 1\n",
        ],
        expected_samples=["3 5\n1 2\n5 11\n9 9\n"],
        small_inputs=[
            "3 2\n7 7 7\n1 3 2\n2 3 1\n",
            gen_38(400, 500, 20, 3801),
            gen_38(1400, 1500, 10**9, 3802),
        ],
        large_inputs=[
            gen_38(160000, 160000, 10**9, 3811),
            gen_38(180000, 180000, 500, 3812),
            gen_38(180000, 180000, 10**9, 3813),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1), vals;
    vals.reserve(n);
    for (int i = 1; i <= n; i++) { cin >> a[i]; vals.push_back(a[i]); }
    sort(vals.begin(), vals.end());
    vals.erase(unique(vals.begin(), vals.end()), vals.end());
    int m = (int)vals.size();

    int max_nodes = (n + 5) * 20;
    vector<int> lc(max_nodes, 0), rc(max_nodes, 0), cnt(max_nodes, 0);
    vector<long long> sm(max_nodes, 0);
    int ptr = 0;

    function<int(int, int, int, int, long long)> upd =
        [&](int prev, int l, int r, int pos, long long v) -> int {
            int u = ++ptr;
            lc[u] = lc[prev];
            rc[u] = rc[prev];
            cnt[u] = cnt[prev] + 1;
            sm[u] = sm[prev] + v;
            if (l == r) return u;
            int mid = (l + r) >> 1;
            if (pos <= mid) lc[u] = upd(lc[prev], l, mid, pos, v);
            else rc[u] = upd(rc[prev], mid + 1, r, pos, v);
            return u;
        };

    vector<int> root(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        int id = (int)(lower_bound(vals.begin(), vals.end(), a[i]) - vals.begin());
        root[i] = upd(root[i - 1], 0, m - 1, id, a[i]);
    }

    while (q--) {
        int l, r, k;
        cin >> l >> r >> k;
        int uL = root[l - 1], uR = root[r];
        int lo = 0, hi = m - 1;
        long long s = 0;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            int cleft = cnt[lc[uR]] - cnt[lc[uL]];
            if (k <= cleft) {
                uL = lc[uL];
                uR = lc[uR];
                hi = mid;
            } else {
                k -= cleft;
                s += sm[lc[uR]] - sm[lc[uL]];
                uL = rc[uL];
                uR = rc[uR];
                lo = mid + 1;
            }
        }
        long long kth = vals[lo];
        s += 1LL * k * kth;
        cout << kth << ' ' << s << '\n';
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
    vector<long long> a(n + 1), buf;
    for (int i = 1; i <= n; i++) cin >> a[i];
    while (q--) {
        int l, r, k;
        cin >> l >> r >> k;
        buf.assign(a.begin() + l, a.begin() + r + 1);
        nth_element(buf.begin(), buf.begin() + (k - 1), buf.end());
        long long kth = buf[k - 1], sum = 0;
        for (int i = 0; i < k; i++) sum += buf[i];
        cout << kth << ' ' << sum << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_39: GUILD TRIBUTES (2000) - Mo's Algorithm (sum of f(x)^2 * x)
    # ------------------------------------------------------------------
    def gen_39(n: int, q: int, vmax: int, seed: int) -> str:
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
        pid="synth_39",
        title="Guild Tributes",
        rating=2000,
        tags=["data structures", "sqrt decomposition", "mo algorithm", "offline"],
        notes="Offline snake-order Mo's algorithm maintaining cnt[x] and delta (2*cnt[x]+1)*x in O((N + Q) * sqrt(N)).",
        story_and_spec="""A street of N guildhalls is numbered 1 to N, where guildhall i belongs to trade tier A_i. For a contiguous block of guildhalls [L, R], if tier x appears f(x) times among A_L, A_{L+1}, ..., A_R, that tier pays f(x)^2 * x gold in tribute.

Given Q queries [L, R], compute the total tribute summed over all tiers present in [L, R].

Input
The first line contains N and Q.
The second line contains A_1 ... A_N.
Each of the next Q lines contains L and R.

Output
For each query print the total tribute on its own line.

Constraints
1 <= N, Q <= 180000
1 <= A_i <= 1000000
1 <= L <= R <= N""",
        samples=[
            "6 3\n1 2 1 3 2 1\n1 4\n2 6\n1 6\n",
        ],
        expected_samples=["9\n15\n20\n"],
        small_inputs=[
            "1 2\n100\n1 1\n1 1\n",
            gen_39(400, 500, 30, 3901),
            gen_39(1500, 1500, 1000000, 3902),
        ],
        large_inputs=[
            gen_39(160000, 160000, 50000, 3911),
            gen_39(180000, 180000, 1000000, 3912),
            gen_39(180000, 180000, 500, 3913),
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
    int mx = 0;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }

    int B = max(1, (int)(n / max(1.0, sqrt((double)q))));
    struct Qry { int l, r, id, blk; };
    vector<Qry> qs(q);
    for (int i = 0; i < q; i++) {
        cin >> qs[i].l >> qs[i].r;
        qs[i].id = i;
        qs[i].blk = qs[i].l / B;
    }
    sort(qs.begin(), qs.end(), [](const Qry& x, const Qry& y) {
        if (x.blk != y.blk) return x.blk < y.blk;
        return (x.blk & 1) ? (x.r > y.r) : (x.r < y.r);
    });

    vector<int> cnt(mx + 1, 0);
    long long cur = 0;
    auto add = [&](int idx) {
        int x = a[idx];
        cur += 1LL * (2 * cnt[x] + 1) * x;
        cnt[x]++;
    };
    auto rem = [&](int idx) {
        int x = a[idx];
        cnt[x]--;
        cur -= 1LL * (2 * cnt[x] + 1) * x;
    };

    vector<long long> ans(q);
    int L = 1, R = 0;
    for (const auto& qr : qs) {
        while (R < qr.r) add(++R);
        while (L > qr.l) add(--L);
        while (R > qr.r) rem(R--);
        while (L < qr.l) rem(L++);
        ans[qr.id] = cur;
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
    int mx = 0;
    for (int i = 1; i <= n; i++) { cin >> a[i]; mx = max(mx, a[i]); }
    vector<int> cnt(mx + 1, 0), touched;
    while (q--) {
        int l, r; cin >> l >> r;
        touched.clear();
        for (int i = l; i <= r; i++) {
            if (!cnt[a[i]]) touched.push_back(a[i]);
            cnt[a[i]]++;
        }
        long long s = 0;
        for (int x : touched) {
            s += 1LL * cnt[x] * cnt[x] * x;
            cnt[x] = 0;
        }
        cout << s << '\n';
    }
    return 0;
}
""",
    ))

    # ------------------------------------------------------------------
    # synth_40: TOKEN DUEL (2200) - Sprague-Grundy on DAG + Winning Move Count
    # ------------------------------------------------------------------
    def gen_40(n: int, m: int, q: int, tot_k: int, seed: int, star_hub: bool = False) -> str:
        r = random.Random(seed)
        edges = []
        if star_hub:
            # Node 2 -> n (so g[2]=1), Node 1 -> 2 and 1 -> 3..n (so g[1]=2, out-degree ~m)
            edges.append((2, n))
            edges.append((1, 2))
            for v in range(3, n + 1):
                edges.append((1, v))
            while len(edges) < m:
                edges.append((1, r.randint(3, n)))
        else:
            for _ in range(m):
                u = r.randint(1, n - 1)
                v = r.randint(u + 1, n)
                edges.append((u, v))
        lines = [f"{n} {len(edges)} {q}"]
        for u, v in edges:
            lines.append(f"{u} {v}")
        rem = tot_k
        for i in range(q):
            left = q - i
            k = max(2, rem - 2 * (left - 1) if i == q - 1 else r.randint(2, max(2, (2 * rem) // left)))
            rem = max(2 * (left - 1), rem - k)
            if star_hub:
                # [1, 1, ..., 2] has XOR != 0 and scans adj[1] (size ~200k) on every token at 1!
                toks = [1] * (k - 1) + [2]
            else:
                toks = [r.randint(1, n) for _ in range(k)]
            lines.append(f"{k} " + " ".join(map(str, toks)))
        return "\n".join(lines) + "\n"

    specs.append(ProblemSpec(
        pid="synth_40",
        title="Token Duel",
        rating=2200,
        tags=["games", "sprague-grundy", "graphs", "dag", "dp"],
        notes="Compute Grundy values g[u] = mex{g[v]} in reverse topological order and compress neighbor Grundy frequencies per vertex so counting winning moves X ^ g[P_j] == g[v] is O(log deg).",
        story_and_spec="""A dungeon has N chambers numbered 1 to N and M one-way tunnels forming a directed acyclic graph (if a tunnel goes from U to V, U != V and there are no directed cycles; parallel tunnels between the same U and V may exist and count as distinct moves).

You evaluate Q independent two-player games. Game i starts with K_i indistinguishable tokens placed at chambers P_1, ..., P_{K_i} (multiple tokens may occupy the same chamber; each of the K_i tokens is treated as a separate piece, so moving token a from U along a tunnel and moving token b from U along the same tunnel are counted as distinct opening moves).

Players First and Second alternate turns, with First moving first. On a turn, a player chooses ONE of the K_i tokens and slides it along ONE outgoing tunnel from its current chamber; a player with no legal move loses.

Under optimal play, determine who wins AND, if First wins, count how many distinct legal opening moves leave Second in a losing position (if Second wins, that count is 0).

Input
The first line contains N, M, and Q.
Each of the next M lines contains U_j and V_j representing a one-way tunnel from U_j to V_j.
Each of the next Q lines begins with K_i followed by P_1 ... P_{K_i}.
The sum of K_i over all games does not exceed 250000.

Output
For each game print either "FIRST W" (where W >= 1 is the number of winning opening moves) or "SECOND 0" on its own line.

Constraints
2 <= N <= 180000
1 <= M <= 220000
1 <= Q <= 120000
1 <= K_i <= 250000, sum of K_i <= 250000""",
        samples=[
            "4 4 3\n1 2\n1 3\n2 4\n3 4\n1 1\n2 2 3\n2 1 2\n",
        ],
        expected_samples=["SECOND 0\nSECOND 0\nFIRST 3\n"],
        small_inputs=[
            "2 2 2\n1 2\n1 2\n1 1\n2 1 1\n",
            gen_40(200, 500, 250, 800, 4001, False),
            gen_40(500, 1200, 400, 1500, 4002, True),
        ],
        large_inputs=[
            gen_40(150000, 200000, 100000, 240000, 4011, True),
            gen_40(170000, 210000, 110000, 240000, 4012, True),
            gen_40(170000, 210000, 100000, 240000, 4013, False),
        ],
        fast_cpp=r"""
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<int>> adj(n + 1);
    vector<int> indeg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        indeg[v]++;
    }

    vector<int> topo;
    topo.reserve(n);
    for (int i = 1; i <= n; i++) if (!indeg[i]) topo.push_back(i);
    for (size_t i = 0; i < topo.size(); i++) {
        int u = topo[i];
        for (int v : adj[u]) if (--indeg[v] == 0) topo.push_back(v);
    }

    vector<int> g(n + 1, 0), seen(n + 2, 0);
    vector<vector<pair<int, int>>> nfreq(n + 1);
    for (int i = n - 1; i >= 0; i--) {
        int u = topo[i];
        vector<int> vals;
        vals.reserve(adj[u].size());
        for (int v : adj[u]) {
            seen[g[v]] = u;
            vals.push_back(g[v]);
        }
        int mex = 0;
        while (seen[mex] == u) mex++;
        g[u] = mex;
        sort(vals.begin(), vals.end());
        for (int x : vals) {
            if (nfreq[u].empty() || nfreq[u].back().first != x) nfreq[u].push_back({x, 1});
            else nfreq[u].back().second++;
        }
    }

    while (q--) {
        int k;
        cin >> k;
        vector<int> p(k);
        int X = 0;
        for (int i = 0; i < k; i++) {
            cin >> p[i];
            X ^= g[p[i]];
        }
        if (X == 0) {
            cout << "SECOND 0\n";
            continue;
        }
        long long ways = 0;
        for (int u : p) {
            int need = X ^ g[u];
            const auto& fv = nfreq[u];
            auto it = lower_bound(fv.begin(), fv.end(), make_pair(need, 0));
            if (it != fv.end() && it->first == need) ways += it->second;
        }
        cout << "FIRST " << ways << '\n';
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
    vector<vector<int>> adj(n + 1);
    vector<int> indeg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        indeg[v]++;
    }
    vector<int> topo;
    for (int i = 1; i <= n; i++) if (!indeg[i]) topo.push_back(i);
    for (size_t i = 0; i < topo.size(); i++) {
        int u = topo[i];
        for (int v : adj[u]) if (--indeg[v] == 0) topo.push_back(v);
    }
    vector<int> g(n + 1, 0), seen(n + 2, 0);
    for (int i = n - 1; i >= 0; i--) {
        int u = topo[i];
        for (int v : adj[u]) seen[g[v]] = u;
        int mex = 0;
        while (seen[mex] == u) mex++;
        g[u] = mex;
    }
    while (q--) {
        int k; cin >> k;
        vector<int> p(k);
        int X = 0;
        for (int i = 0; i < k; i++) { cin >> p[i]; X ^= g[p[i]]; }
        if (X == 0) { cout << "SECOND 0\n"; continue; }
        long long ways = 0;
        for (int u : p) {
            int need = X ^ g[u];
            for (int v : adj[u]) if (g[v] == need) ways++;
        }
        cout << "FIRST " << ways << '\n';
    }
    return 0;
}
""",
    ))

    return specs


if __name__ == "__main__":
    build_and_verify_batch(make_batch_08())
